import numpy as np
import random
from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
# import bezier
import numpy as np
from kornia.geometry.transform import get_tps_transform, warp_image_tps
class HybridAugmentor(nn.Module):
    def __init__(self, num_classes, tau_max=0.7, tau_min=0.3, gamma=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.gamma = gamma
        self.controller = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )  # Progressive augmentation controller

    # def class_aware_mixup(self, x, masks, alpha=0.4):
    #     """Class-aware nonlinear mixup augmentation"""
    #     batch_size = x.size(0)
    #     lam = np.random.beta(alpha, alpha)
        
    #     # Generate mixed image using class-specific masks
    #     perm = torch.randperm(batch_size)
    #     mixed_x = torch.zeros_like(x)
    #     for c in range(self.num_classes):
    #         mask = masks[:,c].unsqueeze(1)
    #         print("Shape of X is :",x.shape)
    #         print("shape of mask  is : ", mask.shape)
    #         mixed_x += mask * (lam * x + (1-lam) * x[perm]) + (1-mask) * x
            
    #     return mixed_x, lam
    
    def class_aware_mixup(self, x, masks, alpha=0.4):
        """
        Class-aware Mixup for multi-class foreground and background handling.

        Parameters:
        - x: Input tensor [batch_size, 1, 192, 192]
        - masks: Class mask [batch_size, 5, 192, 192] (0 for background)
        - alpha: Beta distribution parameter for λ

        Returns:
        - mixed_x: Mixed image
        - lam: Mixing coefficient
        """
        batch_size, _, height, width = x.shape
        num_classes = masks.shape[1]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Shuffle for mixup
        perm = torch.randperm(batch_size)

        # Ensure mask size matches input
        masks = masks[:, :, :height, :width]

        # Initialize output
        mixed_x = torch.zeros_like(x)

        # Handle background (mask == 0)
        background_mask = (masks.sum(dim=1, keepdim=True) == 0)
        mixed_x += background_mask * x

        # Iterate through each foreground class (mask > 0)
        for c in range(1, num_classes + 1):
            # Identify pixels belonging to class 'c'
            class_mask = (masks[:, c - 1] == 1).unsqueeze(1)  # Shape: [32, 1, 192, 192]
            # print("Shape of class mask: ",class_mask.shape)
            # print("Shape of x : ",x.shape)
            # Skip if no pixels for this class
            if class_mask.sum() == 0:
                continue

            # Apply class-aware mixup
            mixed_x += class_mask * (lam * x + (1 - lam) * x[perm])

        return mixed_x, lam
        
    def bezier_transform(self, x, masks, control_points=3):
        """
        Class-specific Bézier curve transformation without using the bezier library.
        Uses grid sampling to achieve per-pixel deformations similar to the original.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            masks: Class-specific masks of shape [B, num_classes, H, W]
            control_points: Number of control points for the Bézier curve
        
        Returns:
            Transformed tensor of the same shape as input
        """
        import torch.nn.functional as F
        
        B, C, H, W = x.shape
        transformed = torch.zeros_like(x)
        
        # Bernstein polynomial basis function
        def bernstein_poly(i, n, t):
            """
            Bernstein polynomial of degree n, term i.
            
            Args:
                i: Index of the term
                n: Degree of polynomial
                t: Parameter value (0 to 1)
                
            Returns:
                Value of the Bernstein polynomial at t
            """
            from math import comb
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        
        # Evaluate Bezier curve for a specific t value
        def bezier_curve_point(t, control_points_array):
            """
            Calculate point on Bezier curve for parameter t.
            
            Args:
                t: Parameter value (0 to 1)
                control_points_array: Control points array of shape [2, n_points]
                
            Returns:
                Point coordinates [x, y]
            """
            n = control_points_array.shape[1] - 1
            point = np.zeros(2)
            
            for i in range(n + 1):
                coef = bernstein_poly(i, n, t)
                point[0] += control_points_array[0, i] * coef
                point[1] += control_points_array[1, i] * coef
                
            return point
        
        # Create normalized sampling grid (from -1 to 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=2).to(x.device)
        
        for c in range(self.num_classes):
            mask = masks[:, c].unsqueeze(1)
            
            # Generate random control points
            nodes = np.random.uniform(0, 1, (2, control_points))
            
            # Create t values corresponding to flattened image pixels
            t_values = torch.linspace(0, 1, H*W)
            
            # Evaluate Bezier curve for each t value (using numpy for simplicity)
            curve_points = np.zeros((H*W, 2))
            for i, t in enumerate(t_values.numpy()):
                curve_points[i] = bezier_curve_point(t, nodes)
            
            # Reshape to match image dimensions
            curve_points = curve_points.reshape(H, W, 2)
            
            # Convert to displacement field (scaled to reasonable values)
            # Normalize to range -0.2 to 0.2 for displacement
            displacement_field = torch.from_numpy(curve_points).float().to(x.device) - 0.5
            displacement_field = displacement_field * 0.4  # Scale factor to control deformation strength
            
            # Apply the displacement to base grid
            sampling_grid = base_grid.clone().repeat(B, 1, 1, 1)
            sampling_grid += displacement_field.unsqueeze(0).repeat(B, 1, 1, 1)
            
            # Use grid_sample to apply the deformation
            warped = F.grid_sample(
                x, 
                sampling_grid,
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
            
            transformed += mask * warped
        
        return transformed
    
    def thin_plate_spline_transform(self, x, masks, control_points=9):
        B, C, H, W = x.shape
        device = x.device
        
        # Generate control points in grid pattern
        n_per_side = int(np.sqrt(control_points))
        points_src = torch.zeros(B, n_per_side**2, 2, device=device)
        points_idx = 0
        for i in range(n_per_side):
            for j in range(n_per_side):
                points_src[:, points_idx, 0] = 2 * i / (n_per_side - 1) - 1  # Range [-1, 1]
                points_src[:, points_idx, 1] = 2 * j / (n_per_side - 1) - 1
                points_idx += 1
        
        # Generate anatomically plausible displacements
        # Different organs have different elasticity - use masks to adjust displacement
        # Create organ-specific displacement maps
        displacements = torch.zeros(B, n_per_side**2, 2, device=device)
        
        # Anatomical constraints (based on common abdominal organ elasticity)
        # Values based on medical literature for abdominal organs
        elasticity_map = {
            0: 0.05,  # Background - minimal deformation
            1: 0.12,  # Liver - moderate deformation
            2: 0.08,  # Spleen - less deformation
            3: 0.15,  # Kidney - more deformation
            4: 0.10,  # Default for other organs
        }
        
        # Get majority organ per control point region to determine deformation magnitude
        for b in range(B):
            for i in range(n_per_side):
                for j in range(n_per_side):
                    idx = i * n_per_side + j
                    
                    # Determine which part of the image this control point affects most
                    # by creating a simple gaussian mask around the control point
                    y_coord = int((i / (n_per_side - 1)) * (H - 1))
                    x_coord = int((j / (n_per_side - 1)) * (W - 1))
                    
                    # Sample 5x5 region around point to determine dominant organ
                    y_min, y_max = max(0, y_coord-2), min(H, y_coord+3)
                    x_min, x_max = max(0, x_coord-2), min(W, x_coord+3)
                    
                    # Get region from mask
                    region = masks[b, :, y_min:y_max, x_min:x_max].sum(dim=(1,2))
                    dominant_organ = torch.argmax(region).item()
                    
                    # Get elasticity factor for this organ
                    elasticity = elasticity_map.get(dominant_organ, 0.10)
                    
                    # Generate random displacement scaled by elasticity
                    displacements[b, idx, 0] = torch.randn(1, device=device) * elasticity
                    displacements[b, idx, 1] = torch.randn(1, device=device) * elasticity
        
        # Apply smoothness constraint to prevent unrealistic deformations
        # Ensure neighboring control points have similar displacements
        smoothed_displacements = displacements.clone()
        for i in range(1, n_per_side-1):
            for j in range(1, n_per_side-1):
                idx = i * n_per_side + j
                # Average with neighbors for smoothness
                neighbors_idx = [
                    (i-1)*n_per_side + j, # top
                    (i+1)*n_per_side + j, # bottom
                    i*n_per_side + (j-1), # left
                    i*n_per_side + (j+1)  # right
                ]
                for b in range(B):
                    # Apply smoothing (80% original + 20% neighbors average)
                    neighbor_avg = torch.stack([displacements[b, n_idx] for n_idx in neighbors_idx]).mean(dim=0)
                    smoothed_displacements[b, idx] = 0.8 * displacements[b, idx] + 0.2 * neighbor_avg
        
        points_dst = points_src + smoothed_displacements
        
        # Use the correct function from kornia
        # from kornia.geometry.transform import thin_plate_spline
        # grid = thin_plate_spline.thin_plate_spline(points_src, points_dst, (H, W))
        kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)
        warped_image = warp_image_tps(x, points_src, kernel_weights, affine_weights, align_corners = True)
        # Apply the transformation with gradient tracking
        # transformed = F.grid_sample(
        #     x, grid, mode='bilinear', padding_mode='border', align_corners=True
        # )
        
        return warped_image

    def to(self, device):
        self.controller = self.controller.to(device)
        return self
    def adaptive_threshold(self, epoch, total_epochs):
            """Sigmoidal curriculum learning for saliency threshold"""
            t = epoch / total_epochs
            return self.tau_min + (self.tau_max - self.tau_min) * (2/(1 + np.exp(-self.gamma*t)) - 1)

    def forward(self, x, masks, epoch, total_epochs):
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = x.device
            # Phase 1: Class-aware nonlinear mixup
            mixed_x, lam = self.class_aware_mixup(x, masks)
            
            # Phase 2: Location-scale transformation
            global_aug = self.thin_plate_spline_transform(mixed_x, masks)
            local_aug = self.thin_plate_spline_transform(x, masks)
            # global_aug = self.bezier_transform(mixed_x, masks)
            # local_aug = self.bezier_transform(x, masks)
            
            ## Ensure 'global_aug' retains gradients
            # global_aug.requires_grad_(True)

            # Controller-adjusted parameters (keep in computational graph)
            # control_params = self.controller(torch.stack([
            #     torch.tensor(lam, device=device),  # Convert lam to a tensor
            #     torch.tensor(epoch / total_epochs, device=device),
            #     global_aug.mean(),
            #     local_aug.std()
            # ]))

            # alpha_ctrl, beta_ctrl, gamma_ctrl = torch.sigmoid(control_params)

            # Compute saliency map without detaching the graph
            # grad_global = torch.autograd.grad(global_aug.sum(), global_aug, create_graph=True, retain_graph=True)[0]

            # Apply adaptive saliency threshold
            # saliency = (grad_global.abs().mean(1, keepdim=True) > self.adaptive_threshold(epoch, total_epochs)).float()

            # Fused output
            # fused = saliency * global_aug + (1 - saliency) * local_aug
            return global_aug, local_aug
    
class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image=self.non_linear_transformation(image, inverse=False)
        image=self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation(self,image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)

        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image
