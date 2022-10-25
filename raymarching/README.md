# `raymarching`

- Generate a rays_generator: `nrg = NeRF_RayGen(H,W,focal)`
- Get rays: `nrg.get_rays(c2w)`, get two `(batch_size,h,w,3)` means the rays.