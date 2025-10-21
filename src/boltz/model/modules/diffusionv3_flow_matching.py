# modified from the diffusionv2.py file
# the idea is to use the flow matching approach instead of the score based approach
# the intial velocity field is computed from the score model and then the flow is computed by solving the ODE

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials


class ScoreToVelocityConverter:
    """
    Analytically convert score-based diffusion predictions to flow matching velocities.
    Integrated directly into the FlowMatchingDiffusion module.
    """
    
    def __init__(
        self,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        conversion_method='noise_based',
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.conversion_method = conversion_method
        
        print(f"Score-to-velocity converter initialized: {conversion_method} method")
    
    def sigma_to_t(self, sigma):
        """Convert noise level σ to flow time t ∈ [0,1]."""
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma)
        
        log_sigma = torch.log(sigma)
        log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=sigma.device))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=sigma.device))
        
        t = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
        return torch.clamp(t, 0.0, 1.0)
    
    def t_to_sigma(self, t):
        """Convert flow time t to noise level σ."""
        if isinstance(t, (int, float)):
            t = torch.tensor(t)
        
        log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=t.device))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=t.device))
        
        log_sigma = log_sigma_min + t * (log_sigma_max - log_sigma_min)
        return torch.exp(log_sigma)
    
    def convert_score_to_velocity(self, x_t, denoised_coords, sigma):
        """
        Convert score model output to velocity field using noise-based method.
        
        Key insight: Both score and flow models parameterize the same noise ε!
        Score model: x_t = x_0 + σ*ε  →  ε = (x_t - x_0)/σ
        Flow model:  x_t = (1-t)*x_0 + t*ε  →  v = ε - x_0
        """
        # Handle sigma dimensions
        if isinstance(sigma, (int, float)):
            sigma_expanded = sigma
        elif sigma.dim() == 1:
            sigma_expanded = sigma.reshape(-1, 1, 1)
        else:
            sigma_expanded = sigma
        
        # Estimate noise: ε = (x_t - x_0)/σ
        epsilon = (x_t - denoised_coords) / (sigma_expanded + 1e-8)
        
        # Flow matching velocity: v = ε - x_0
        velocity = epsilon - denoised_coords
        
        return velocity


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        conversion_method: str = 'noise_based',  # Method for score-to-velocity conversion
        **kwargs  # Accept any additional parameters for compatibility
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        
        # Initialize score-to-velocity converter
        self.converter = ScoreToVelocityConverter(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            conversion_method=conversion_method,
        )

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # Score model preconditioning functions (needed for analytical conversion)
    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(self, noised_atom_coords, sigma, network_condition_kwargs):
        """Get denoised coordinates from the pre-trained score model."""
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    # old code from diffusionv2.py now going to use flow matching instead
    def compute_flow_path(self, x0, xi, t):
        """
        Compute interpolated position along flow path.
        x0: clean data coordinates [b, m, 3]
        xi: noise sample [b, m, 3]  
        t: time in [0,1], shape [b] or [b,1,1]
        
        Returns x_t = (1-t)*x0 + t*xi
        """
        if t.dim() == 1:
            t = t.reshape(-1,1,1)
        return ( 1 - t ) * x0 + t * xi

    def compute_target_velocity(self, x0, xi, t):
        """
        Compute target velocity field for rectified flow.
        
        For the linear path x_t = (1-t)*x0 + t*xi,
        the velocity is dx_t/dt = -x0 + xi = xi - x0 (constant!).
        
        x0: clean data coordinates [b, m, 3]
        xi: noise sample [b, m, 3]  
        t: time in [0,1], shape [b] or [b,1,1] (not used for rectified flow)
        
        Returns v_t = xi - x0
        """
        return xi - x0


    # new code for flow matching
    def velocity_network_forward(self, atom_coords_t, t, network_condition_kwargs):
        """
        Predict velocity using the probability flow ODE from the pre-trained score model.
        
        This uses the correct EDM probability flow ODE velocity instead of trying 
        to convert to a different flow path.
        """
        batch, device = atom_coords_t.shape[0], atom_coords_t.device
        
        if isinstance(t, float):
            t = torch.full((batch,), t, device=device)
        
        if t.dim() > 1:
            t = t.squeeze()
        
        # Convert flow time t ∈ [0,1] to noise level sigma
        # t=0 → high noise (sigma_max), t=1 → low noise (sigma_min) 
        sigma = self.sigma_max * (1 - t) + self.sigma_min * t
        
        # Get the score model's denoised prediction
        denoised_coords = self.preconditioned_network_forward(
            atom_coords_t, sigma, network_condition_kwargs
        )
        
        # Use the EDM probability flow ODE velocity:
        # v = (x_0_pred - x_t) / sigma
        # This is the correct velocity for the probability flow ODE
        if isinstance(sigma, float):
            sigma_expanded = sigma
        elif sigma.dim() == 1:
            sigma_expanded = sigma.reshape(-1, 1, 1)
        else:
            sigma_expanded = sigma
        
        velocity = (denoised_coords - atom_coords_t) / (sigma_expanded + 1e-8)
        
        return velocity


    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        """
        Sample by integrating the ODE dx/dt = u_theta(x,t) from t=1 to t=0.
        Uses Heun's method (2nd order Runge-Kutta).
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        
        shape = (*atom_mask.shape, 3)
        
        # Start from pure noise at t=1
        x = torch.randn(shape, device=self.device)
        
        # Time discretization: t=1 -> t=0
        t_schedule = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=self.device)
        
        batch_size = x.shape[0]
        
        for i in range(num_sampling_steps):
            t_curr = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_curr  # negative (moving backward in time)
            
            # Convert scalar times to batch tensors
            t_curr_batch = torch.full((batch_size,), t_curr.item(), device=self.device)
            t_next_batch = torch.full((batch_size,), t_next.item(), device=self.device)
            
            # Heun's method (RK2)
            # First evaluation
            with torch.no_grad():
                v1 = self.velocity_network_forward(
                    x, t_curr_batch, network_condition_kwargs
                )
            
            # Euler step
            x_euler = x + dt * v1
            
            # Second evaluation
            with torch.no_grad():
                v2 = self.velocity_network_forward(
                    x_euler, t_next_batch, network_condition_kwargs
                )
            
            # Heun update (average of two velocities)
            x = x + 0.5 * dt * (v1 + v2)
        
        return {"sample_atom_coords": x}

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        """
        Flow matching training forward pass.
        Samples random time t, creates interpolated coordinates x_t,
        and predicts velocity.
        """
        batch_size = feats["coords"].shape[0] // multiplicity
        
        atom_coords = feats["coords"]
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        
        # Center and optionally augment coordinates
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )
        
        # Sample time uniformly from [0,1]
        if self.synchronize_sigmas:
            t = torch.rand(batch_size, device=self.device).repeat_interleave(multiplicity, 0)
        else:
            t = torch.rand(batch_size * multiplicity, device=self.device)
        
        # Sample noise endpoint
        xi = torch.randn_like(atom_coords)
        
        # Compute flow path interpolation: x_t = (1-t)*x0 + t*xi
        atom_coords_t = self.compute_flow_path(atom_coords, xi, t)
        
        # Compute target velocity: v = xi - x0 (constant for rectified flow)
        target_velocity = self.compute_target_velocity(atom_coords, xi, t)
        
        # Predict velocity with network
        predicted_velocity = self.velocity_network_forward(
            atom_coords_t, 
            t,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            }
        )
        
        return {
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
            "t": t,
            "aligned_true_atom_coords": atom_coords,
            "atom_coords_t": atom_coords_t,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        """
        Compute flow matching loss.
        Main loss: MSE between predicted and target velocity.
        Optional: smooth lDDT loss on reconstructed structure.
        """
        with torch.autocast("cuda", enabled=False):
            predicted_velocity = out_dict["predicted_velocity"].float()
            target_velocity = out_dict["target_velocity"].float()
            t = out_dict["t"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            # Compute weights for different molecule types
            align_weights = predicted_velocity.new_ones(predicted_velocity.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            # MSE loss on velocity vectors
            velocity_mse = ((predicted_velocity - target_velocity) ** 2).sum(dim=-1)
            velocity_mse = torch.sum(
                velocity_mse * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # Optional: time-dependent weighting to emphasize precision at small t
            # For now using uniform weighting
            mse_loss = velocity_mse.mean()

            total_loss = mse_loss

            # Optional: add smooth lDDT loss on predicted x0
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                # Reconstruct x0 from velocity: x0 = x_t - t*v
                t_expanded = t.reshape(-1, 1, 1)
                atom_coords_t = out_dict.get("atom_coords_t", None)
                if atom_coords_t is None:
                    # Reconstruct from target if not stored
                    atom_coords_pred = out_dict["aligned_true_atom_coords"]
                else:
                    atom_coords_pred = atom_coords_t - t_expanded * predicted_velocity

                lddt_loss = smooth_lddt_loss(
                    atom_coords_pred,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "velocity_mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}

    def validate_velocity_field(self, feats, diffusion_conditioning, s_inputs, s_trunk, num_samples=10):
        """
        Validate that the velocity field is well-behaved.
        Checks:
        1. Velocity predictions are consistent across different t values
        2. Velocity magnitudes are reasonable
        3. ODE integration doesn't explode
        
        Returns dict with validation metrics.
        """
        self.eval()
        
        atom_coords = feats["coords"]
        validation_metrics = {}
        
        with torch.no_grad():
            # Check velocity at different time points
            t_values = torch.linspace(0.1, 0.9, 5, device=self.device)
            velocities = []
            
            for t_val in t_values:
                t = torch.full((atom_coords.shape[0],), t_val, device=self.device)
                xi = torch.randn_like(atom_coords)
                x_t = self.compute_flow_path(atom_coords, xi, t)
                
                v_pred = self.velocity_network_forward(
                    x_t, t,
                    network_condition_kwargs={
                        "s_inputs": s_inputs,
                        "s_trunk": s_trunk,
                        "feats": feats,
                        "multiplicity": 1,
                        "diffusion_conditioning": diffusion_conditioning,
                    }
                )
                velocities.append(v_pred)
                
                # Check magnitude
                v_norm = torch.norm(v_pred, dim=-1).mean()
                validation_metrics[f"velocity_norm_t{t_val:.1f}"] = v_norm.item()
            
            # Check velocity smoothness across time
            velocity_changes = []
            for i in range(len(velocities) - 1):
                change = torch.norm(velocities[i+1] - velocities[i], dim=-1).mean()
                velocity_changes.append(change.item())
            
            validation_metrics["velocity_temporal_smoothness"] = np.mean(velocity_changes)
            
            # Test ODE integration stability
            try:
                result = self.sample(
                    atom_mask=feats["atom_pad_mask"],
                    num_sampling_steps=10,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                )
                coords_norm = torch.norm(result["sample_atom_coords"], dim=-1).mean()
                validation_metrics["sample_coords_norm"] = coords_norm.item()
                validation_metrics["ode_integration_stable"] = True
            except Exception as e:
                validation_metrics["ode_integration_stable"] = False
                validation_metrics["ode_error"] = str(e)
        
        self.train()
        return validation_metrics

    def benchmark_sampling_speed(
        self,
        feats,
        diffusion_conditioning,
        s_inputs,
        s_trunk,
        step_counts=[5, 10, 20, 40],
        num_runs=3,
    ):
        """
        Benchmark sampling speed for different step counts.
        
        Returns dict mapping step_count -> average time per sample (seconds).
        """
        import time
        
        self.eval()
        benchmark_results = {}
        
        with torch.no_grad():
            for num_steps in step_counts:
                times = []
                
                for _ in range(num_runs):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start = time.time()
                    
                    result = self.sample(
                        atom_mask=feats["atom_pad_mask"],
                        num_sampling_steps=num_steps,
                        s_inputs=s_inputs,
                        s_trunk=s_trunk,
                        feats=feats,
                        diffusion_conditioning=diffusion_conditioning,
                    )
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = time.time() - start
                    times.append(elapsed)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                benchmark_results[f"steps_{num_steps}"] = {
                    "mean_time": avg_time,
                    "std_time": std_time,
                    "samples_per_sec": 1.0 / avg_time,
                }
                
                print(f"Steps: {num_steps:3d} | Time: {avg_time:.3f}s ± {std_time:.3f}s | {1.0/avg_time:.2f} samples/sec")
        
        self.train()
        return benchmark_results
