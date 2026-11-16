use bytemuck::{Pod, Zeroable};

use crate::PhysicsError;

/// Tightly packed camera-major RGBA8 input and camera-major planar RGB output.
///
/// Input byte address: `4 * ((camera * height + y) * width + x) + channel`.
/// Output f32 address: `((camera * 3 + channel) * height + y) * width + x`.
/// RGB channels are scaled by 1/255; alpha is ignored. No gamma transform is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageShape {
    width: u32,
    height: u32,
    cameras: u32,
    pixels_per_image: u32,
    pixels: u32,
}

impl ImageShape {
    pub fn new(width: u32, height: u32, cameras: u32) -> Result<Self, PhysicsError> {
        let pixels_per_image = width
            .checked_mul(height)
            .ok_or(PhysicsError::InvalidImageShape)?;
        let pixels = pixels_per_image
            .checked_mul(cameras)
            .ok_or(PhysicsError::InvalidImageShape)?;
        if pixels == 0 || cameras > i32::MAX as u32 {
            return Err(PhysicsError::InvalidImageShape);
        }
        Ok(Self { width, height, cameras, pixels_per_image, pixels })
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }

    pub const fn cameras(self) -> u32 {
        self.cameras
    }

    pub const fn pixels_per_image(self) -> u32 {
        self.pixels_per_image
    }

    pub const fn pixels(self) -> u32 {
        self.pixels
    }

    pub const fn rgba_bytes(self) -> usize {
        self.pixels as usize * 4
    }

    pub const fn rgb_bytes(self) -> usize {
        self.pixels as usize * 3 * core::mem::size_of::<f32>()
    }
}

/// One result per camera, written on the GPU without atomics or scratch buffers.
/// Luminance uses Rec.709 weights on the stored RGB values (not linearized light).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ImageStatistics {
    /// Mean normalized R, G, B and reserved zero.
    pub mean_rgb: [f32; 4],
    pub mean_luminance: f32,
    pub min_luminance: f32,
    pub max_luminance: f32,
    pub pixel_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn image_storage_sizes_and_statistics_layout() {
        let shape = ImageShape::new(7, 5, 3).unwrap();
        assert_eq!(shape.pixels_per_image(), 35);
        assert_eq!(shape.pixels(), 105);
        assert_eq!(shape.rgba_bytes(), 420);
        assert_eq!(shape.rgb_bytes(), 1260);
        assert_eq!(size_of::<ImageStatistics>(), 32);
        assert_eq!(align_of::<ImageStatistics>(), 16);
        assert_eq!(offset_of!(ImageStatistics, mean_luminance), 16);
        assert_eq!(offset_of!(ImageStatistics, pixel_count), 28);
    }

    #[test]
    fn image_shapes_reject_empty_and_overflowing_sizes() {
        assert!(ImageShape::new(0, 4, 1).is_err());
        assert!(ImageShape::new(4, 4, 0).is_err());
        assert!(ImageShape::new(u32::MAX, 2, 1).is_err());
        assert!(ImageShape::new(1, 1, u32::MAX).is_err());
    }
}
