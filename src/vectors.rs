use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
use core::fmt::Debug;
use defmt::{error, info, Format};
use embedded_graphics::prelude::Point;

use micromath::vector::{Component, Vector, Vector2d};
use num_traits::{self, Float, FromPrimitive, ToPrimitive};

// use defmt::info;
use thiserror_no_std::Error;

#[derive(Debug, Error, defmt::Format)]
pub enum SpriteError {
    ConversionError,
    // ConfigError(#[from] esp_hal::spi::master::ConfigError),
    Infallible(#[from] core::convert::Infallible),
    // SpiError(#[from] esp_hal::spi::Error),
    OtherError,
}

extern crate micromath as mm;

pub trait VecComp: 
    Component
    + ToPrimitive
    + FromPrimitive
    + core::ops::Neg<Output = Self>
    // + SquareRoot
    + defmt::Format
{}

impl VecComp for i32 {}
impl VecComp for f32 {}


#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVector<C: VecComp>(pub Vector2d<C>);

impl Mul<f32> for SpriteVector<f32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let x = self.x * rhs;
        let y = self.y * rhs;
        (x,y).into()
    }
}
impl Mul<f32> for SpriteVector<i32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let x = self.x as f32 * rhs;
        let y = self.y as f32 * rhs;
        (x,y).into()
    }
}
impl Mul<i32> for SpriteVector<i32> {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self::Output {
        let x = self.x * rhs;
        let y = self.y * rhs;
        (x,y).into()
    }
}

impl Sub for SpriteVector<i32> {
    type Output = SpriteVector<i32>;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.x - rhs.x, self.y - rhs.y).into()
    }
}

impl Sub for SpriteVector<f32> {
    type Output = SpriteVector<f32>;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.x - rhs.x, self.y - rhs.y).into()
    }
}

impl Add for SpriteVector<i32> {
    type Output = SpriteVector<i32>;

    fn add(self, rhs: Self) -> Self::Output {
        (self.x + rhs.x, self.y + rhs.y).into()
    }
}
impl Add for SpriteVector<f32> {
    type Output = SpriteVector<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        (self.x + rhs.x, self.y + rhs.y).into()
    }
}
impl Neg for SpriteVector<i32> {
    type Output = SpriteVector<i32>;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}
impl Neg for SpriteVector<f32> {
    type Output = SpriteVector<f32>;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}
impl<C: VecComp> Add for SpriteVector<C> 
where 
    SpriteVector<C>: VectorXY<C>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.get_x() + rhs.get_x();
        let y = self.get_y() + rhs.get_y();
        (x,y).into()
    }

}

impl<C: VecComp> Sub for SpriteVector<C> 
where 
    SpriteVector<C>: VectorXY<C>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let x = self.get_x() - rhs.get_x();
        let y = self.get_y() - rhs.get_y();
        (x,y).into()
    }

}

impl<C: VecComp> From<Point> for SpriteVector<C> {
    fn from(value: Point) -> Self {
        let x = C::from_i32(value.x).unwrap_or_default();
        let y = C::from_i32(value.y).unwrap_or_default();
        Self(Vector2d { x, y })
    }
}

impl From<(f32,f32)> for SpriteVector<i32> {
    fn from(value: (f32,f32)) -> Self {
        let (x,y) = value;
        (x.round() as i32, y.round() as i32).into()
    }
}


impl<C: VecComp> core::ops::DerefMut for SpriteVector<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<C: VecComp> core::ops::Deref for SpriteVector<C> {
    type Target = Vector2d<C>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: VecComp> defmt::Format for SpriteVector<C> {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl<C: VecComp> PartialOrd for SpriteVector<C> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.to_array().partial_cmp(&other.0.to_array())
    }
}


impl<C: VecComp> From<SpriteVector<C>> for Point {
    fn from(value: SpriteVector<C>) -> Self {

        let x = value.0.x.to_f32().unwrap_or_else(||{
            error!("unable to convert from {}", value.0.x);
            panic!();
            f32::default()
        });
        let y = value.0.y.to_f32().unwrap_or_else(||{
            error!("unable to convert from {}", value.0.y);
            panic!();
            f32::default()
        });
        Point { x: x.round() as i32, y: y.round() as i32 }
    }
}
impl<C: VecComp> From<(C,C)> for SpriteVector<C> {
    fn from(value: (C,C)) -> Self {
        Self(value.into())
    }
}

pub trait VectorXY<C: VecComp> {
    fn get_x(&self) -> C;
    fn get_y(&self) -> C;
}

// pub trait SquareRoot {
//     fn calc_sqrt(&self) -> Self;
// }

// impl SquareRoot for i32 {
//     fn calc_sqrt(&self) -> i32 {
//         self.isqrt()
//     }
// }

// impl SquareRoot for f32 {
//     fn calc_sqrt(&self) -> f32 {
//         self.sqrt()
//     }
// }


// impl<C: VecComp> VectorXY<C> for GfxVector<C> {
//     fn get_x(&self) -> C {
//         self.x
//     }

//     fn get_y(&self) -> C {
//         self.y
//     }
// }

// // impl VectorTrait<i32> for GfxVector<i32> {}
// // impl VectorTrait<f32> for GfxVector<f32> {}

// impl<C: VecComp> GfxVector<C> {
//     /// get the magnitude of the vector in units C 
//     fn magnitude(&self) -> C {
//         let m = (self.get_x() * self.get_x()) + (self.get_y() * self.get_y());
//         m.calc_sqrt()
//     }

//     /// calculate the dot product of this vector and another one
//     fn dot_product(&self, other: &Self) -> C {
//         (self.get_x() * other.get_x()) + (self.get_y() * other.get_y())
//     }

//     /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
//     fn normalize(&self) -> Self {
//         let mag =  self.magnitude().to_i32().unwrap_or_default();
//         if mag == 0 {
//             panic!("divide by zero!");
//         } else {
//             let x = self.get_x() / C::from_i32(mag).unwrap();
//             let y = self.get_y() / C::from_i32(mag).unwrap();
//             (x, y).into()
//         }
//     }

//     /// rotate the vector 90 degrees
//     fn rotate90(self) -> Self {
//         let x = -self.get_y();
//         let y = self.get_x();
//         (x, y).into()
//     }

//     /// Calculate the component of the incoming velocity perpendicular to the collision surface
//     fn perpendicular_velocity(&self, collision_normal_normalized: &Self) -> Self {
//         let normal = collision_normal_normalized;
//         let dot = self.dot_product(normal);
//         *normal * dot
//     }


//     /// Calculate the component of the incoming velocity parallel to the collision surface
//     fn parallel_velocity(&self, perpendicular_velocity: &Self) -> Self {
//         self - perpendicular_velocity
//     }


//     fn calculate_reflection_vector_cor(&self, collision_normal: &Self, coefficient_of_restitution: f32) -> Self {
//         // Ensure the normal is normalized (unit length)
//         let normal = collision_normal.normalize();

//         // Calculate the component of the incoming velocity perpendicular to the collision surface
//         let perpendicular_velocity = self.perpendicular_velocity(&normal);

//         // Calculate the component of the incoming velocity parallel to the collision surface
//         let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

//         // The reflected perpendicular velocity is reversed and scaled by the COR
//         let reflected_perpendicular_velocity = coefficient_of_restitution * (-perpendicular_velocity);

//         // The reflected velocity is the sum of the reflected perpendicular and parallel components
//         reflected_perpendicular_velocity + parallel_velocity

//     }
// }

// impl GfxVector<i32> {
//     pub fn calculate_reflection_vector(&self, collision_normal: &Self) -> Self {
//         // Ensure the normal is normalized (unit length)
//         let normal = collision_normal.normalize();

//         // Calculate the component of the incoming velocity perpendicular to the collision surface
//         let perpendicular_velocity = self.perpendicular_velocity(&normal);

//         // Calculate the component of the incoming velocity parallel to the collision surface
//         let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

//         // The reflected perpendicular velocity is reversed and scaled by the COR
//         let reflected_perpendicular_velocity = -perpendicular_velocity;

//         // The reflected velocity is the sum of the reflected perpendicular and parallel components
//         reflected_perpendicular_velocity + parallel_velocity
//     }
// }

// impl GfxVector<f32> {
//     pub fn calculate_reflection_vector(&self, collision_normal: &Self) -> Self {
//         // Ensure the normal is normalized (unit length)
//         let normal = collision_normal.normalize();

//         // Calculate the component of the incoming velocity perpendicular to the collision surface
//         let perpendicular_velocity = self.perpendicular_velocity(&normal);

//         // Calculate the component of the incoming velocity parallel to the collision surface
//         let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

//         // The reflected perpendicular velocity is reversed and scaled by the COR
//         let reflected_perpendicular_velocity = -perpendicular_velocity;

//         // The reflected velocity is the sum of the reflected perpendicular and parallel components
//         reflected_perpendicular_velocity + parallel_velocity
//     }
// }

impl<C: VecComp>  SpriteVector<C> {
    pub fn new(x: C, y: C) -> Self {
        Self(Vector2d { x, y })
    }
}

impl<C: VecComp> From<Vector2d<C>> for SpriteVector<C> {
    fn from(value: Vector2d<C>) -> Self {
        SpriteVector(value)
    }
}


pub trait VecNormalize {
    fn normalize(&self) -> Self;
}

// impl<T> VecNormalize for Vector2d<T> 
// where 
//     T: micromath::vector::Component + core::convert::From<f32>,
//     f32: core::convert::From<T>,
// {
//     fn normalize(&self) -> Self {
//         let mag = self.magnitude();
//         let x = <f32 as core::convert::From<T>>::from(self.x) / mag;
//         let y = <f32 as core::convert::From<T>>::from(self.y) / mag;
//         Vector2d { x: x.into(), y: y.into()}
//     }
// }


impl VecNormalize for Point {
    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    /// if we would divide by zero return a vector of 0,0
    fn normalize(&self) -> Self {
        let mag = (self.x.pow(2) + self.y.pow(2)).isqrt();
        if mag == 0 {
            Self::zero()
        } else {
            Self { x: self.x / mag, y: self.y / mag }
        }
    }
}


fn normalize_vector2d_f32(a_vec: micromath::vector::Vector2d<f32>) -> micromath::vector::Vector2d<f32> {
    let mag = micromath::vector::Vector::magnitude(a_vec);
    micromath::vector::Vector2d { x: a_vec.x / mag, y: a_vec.y / mag }
}

// fn calculate_reflection_vector<T>(
//     incoming_velocity: &Vector2d<T>,
//     collision_normal: &Vector2d<T>,
//     coefficient_of_restitution: f32,
// ) -> Vector2d<T> 
// where 
//     Vector2d<T>: VecNormalize + Mul<f32, Output = Vector2d<T>> + Mul<T, Output = Vector2d<T>>,
//     T: micromath::vector::Component + core::ops::Neg<Output = T>, 
//     <T as Neg>::Output: Mul<f32>, 
//     T: From<f32> + Mul<f32, Output = T>, 
// {
//     // Ensure the normal is normalized (unit length)
//     // let mag = collision_normal.magnitude()
//     let normal: Vector2d<T> = collision_normal.normalize();

//     // Calculate the component of the incoming velocity perpendicular to the collision surface
//     let perpendicular_velocity: Vector2d<T> = normal * incoming_velocity.dot(normal);
//     let neg_x = -perpendicular_velocity.x;
//     let neg_y = -perpendicular_velocity.y;
//     let neg_perpendicular_velocity: Vector2d<T> = Vector2d { x: neg_x , y: neg_y };

//     // Calculate the component of the incoming velocity parallel to the collision surface
//     let parallel_velocity: Vector2d<T> = *incoming_velocity - perpendicular_velocity;

//     // The reflected perpendicular velocity is reversed and scaled by the COR
//     let reflected_perpendicular_velocity: Vector2d<T> = neg_perpendicular_velocity * coefficient_of_restitution;

//     // The reflected velocity is the sum of the reflected perpendicular and parallel components
//     reflected_perpendicular_velocity + parallel_velocity
// }    

// pub fn calculate_reflection_vector_p(
//     incoming_velocity: &Point,
//     collision_normal: &Point,
//     coefficient_of_restitution: f32,
// ) -> Point {
//     let incoming_velocity = GfxVector::<i32>::from(incoming_velocity);
//     let collision_normal = GfxVector::<i32>::from(collision_normal);
//     let reflection = GfxVector::calculate_reflection_vector_cor(&incoming_velocity, &collision_normal, coefficient_of_restitution);
//     reflection.into()
// }    


pub trait VectorLike2d<C> {
    fn get_x(self) -> C;
    fn get_y(self) -> C;

    /// calculate the dot product of two vectors
    fn dot(self, rhs: Self) -> C;

    /// Compute the magnitude of a vector
    fn magnitude(self) -> f32;

    /// Compute the distance between two vectors
    fn distance(self, rhs: Self) -> f32;

}


pub struct Coefficient(pub f32);


pub trait Velocity<C>: VectorLike2d<C> + Copy + Clone + PartialEq + defmt::Format
where 
    Self: From<Point> + From<(C,C)> + Mul<Coefficient, Output = Self>,
    C: VecComp,
{
    #[inline]
    fn speed(&self) -> f32 {
        self.magnitude()
    }

    #[inline]
    fn direction(&self) -> Self {
        self.normalize()
    }

    fn neg(self) -> Self {
        let x = -self.get_x();
        let y = -self.get_y();
        (x,y).into()
    }

    #[inline]
    fn zero() -> Self {
        (C::default(), C::default()).into()
    }

    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    /// if we would divide by zero return a vector of 0,0
    fn normalize(&self) -> Self {
        let mag =  self.magnitude();
        if mag as i32 == 0 {
            Self::zero()
        } else {
            let x = self.get_x().to_f32().unwrap_or(0.0) / mag;
            let y = self.get_y().to_f32().unwrap_or(0.0) / mag;
            (C::from_f32(x).unwrap_or_default(), C::from_f32(y).unwrap_or_default()).into()
        }
    }

    /// rotate the vector 90 degrees
    fn rotate90(self) -> Self {
        let x = -self.get_y();
        let y = self.get_x();
        (x, y).into()
    }

    fn mul_c(&self, c: C) -> Self {
        let x = self.get_x() * c;
        let y = self.get_y() * c;
        (x,y).into()
    }

    /// Calculate the component of the incoming velocity perpendicular to the collision surface
    fn perpendicular_velocity(&self, collision_normal_normalized: &Self) -> Self {
        let normal = collision_normal_normalized;
        let dot = self.dot(*normal);
        normal.mul_c(dot)
    }


    fn sub(&self, rhs: &Self) -> Self {
        let x = self.get_x() - rhs.get_x();
        let y = self.get_y() - rhs.get_y();
        (x,y).into()
    }

    fn add_assign(&mut self, rhs: &Self) -> Self {
        let x = self.get_x() + rhs.get_x();
        let y = self.get_y() + rhs.get_y();
        *self = (x,y).into();
        *self
    }
    
    /// Calculate the component of the incoming velocity parallel to the collision surface
    fn parallel_velocity(&self, perpendicular_velocity: &Self) -> Self {
        self.sub(perpendicular_velocity)
    }

    fn calculate_reflection_vector(&self, collision_normal: &Self) -> Self {
        // let speed = self.magnitude();
        {
            // Ensure the normal is normalized (unit length)
            let normal = collision_normal.normalize();

            // Calculate the component of the incoming velocity perpendicular to the collision surface
            let perpendicular_velocity = self.perpendicular_velocity(&normal);

            // Calculate the component of the incoming velocity parallel to the collision surface
            let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

            // The reflected perpendicular velocity is reversed and scaled by the COR
            let mut reflected_perpendicular_velocity = perpendicular_velocity.neg();

            // The reflected velocity is the sum of the reflected perpendicular and parallel components
            reflected_perpendicular_velocity.add_assign(&parallel_velocity);

            if *self == reflected_perpendicular_velocity {
                info!("velocity = {}", self);
                info!("perpendicular_velocity = {}", perpendicular_velocity);
                info!("parallel_velocity = {}", parallel_velocity);
                info!("reflected_perpendicular_velocity = {}", reflected_perpendicular_velocity);
            }
            reflected_perpendicular_velocity
        }
        // let speed2 = reflection.magnitude();
        // if (speed - speed2).abs() >= 3.0*f32::EPSILON {
        //     info!("speed changed by {}", (speed - speed2));
        // }
        // let ref_norm = reflection.normalize();
        // ref_norm * Coefficient(speed)
        // reflection
    }

    fn calculate_reflection_vector_cor(&self, collision_normal: &Self, coefficient_of_restitution: Coefficient) -> Self {
        // Ensure the normal is normalized (unit length)
        let normal = collision_normal.normalize();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        let perpendicular_velocity = self.perpendicular_velocity(&normal);

        // Calculate the component of the incoming velocity parallel to the collision surface
        let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let mut reflected_perpendicular_velocity = perpendicular_velocity.neg();
        if coefficient_of_restitution.0 == 1.0 {
            reflected_perpendicular_velocity = reflected_perpendicular_velocity * coefficient_of_restitution;
        }

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        reflected_perpendicular_velocity.add_assign(&parallel_velocity)

    }
}


#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VelocityCalculationResults<T: VecComp> {
    pub initial: SpriteVector<T>,
    pub collision_normal: SpriteVector<T>,
    pub normal: SpriteVector<T>,
    pub perpendicular: SpriteVector<T>,
    pub parallel: SpriteVector<T>,
    pub reflected: SpriteVector<T>,
}

impl<T: VecComp> defmt::Format for VelocityCalculationResults<T> {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "vcr: i:{} cn:{} n:{} pe:{} pa:{} r:{}",
        self.initial,
        self.collision_normal,
        self.normal,
        self.perpendicular,
        self.parallel,
        self.reflected)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorFloat(pub Vector2d<f32>);

impl PartialOrd for SpriteVectorFloat {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.to_array().partial_cmp(&other.0.to_array())
    }
}

impl From<SpriteVectorFloat> for SpriteVector<i32> {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x, value.y).into()
    }
}

impl From<SpriteVectorFloat> for SpriteVector<f32> {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x, value.y).into()
    }
}

impl From<(f32,f32)> for SpriteVectorFloat {
    fn from(value: (f32,f32)) -> Self {
        Self(value.into())
    }
}

impl From<Point> for SpriteVectorFloat {
    fn from(value: Point) -> Self {
        Self((value.x as f32, value.y as f32).into())
    }
}

impl From<SpriteVectorFloat> for Point {
    fn from(value: SpriteVectorFloat) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}
impl core::ops::Deref for SpriteVectorFloat {
    type Target = Vector2d<f32>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for SpriteVectorFloat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl defmt::Format for SpriteVectorFloat {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl AddAssign for SpriteVectorFloat {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl From<(i32,i32)> for SpriteVectorFloat {
    fn from(value: (i32,i32)) -> Self {
        (value.0 as f32, value.1 as f32).into()
    }
}

impl Neg for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn neg(self) -> Self::Output {
        (-self.x, -self.y).into()
    }
}

impl Sub for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.x - rhs.x, self.y - rhs.y).into()
    }
}

impl Add for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn add(self, rhs: Self) -> Self::Output {
        (self.x + rhs.x, self.y + rhs.y).into()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpriteVectorInt(pub Vector2d<i32>);


impl From<(i32,i32)> for SpriteVectorInt {
    fn from(value: (i32,i32)) -> Self {
        Self(value.into())
    }
}

impl From<Point> for SpriteVectorInt {
    fn from(value: Point) -> Self {
        Self((value.x, value.y).into())
    }
}

impl core::ops::DerefMut for SpriteVectorInt {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl core::ops::Deref for SpriteVectorInt {
    type Target = Vector2d<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl defmt::Format for SpriteVectorInt {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.0.x, self.0.y)
    }
}

impl From<SpriteVectorInt> for SpriteVectorFloat {
    fn from(value: SpriteVectorInt) -> Self {
        (value.x as f32, value.y as f32).into()
    }
}
impl From<SpriteVectorFloat> for SpriteVectorInt {
    fn from(value: SpriteVectorFloat) -> Self {
        (value.x.round() as i32, value.y.round() as i32).into()
    }
}

// impl Velocity<i32> for SpriteVectorInt {
//     fn zero() -> Self {
//         (0,0).into()
//     }
// }

// impl Velocity<f32> for SpriteVectorFloat {
//     fn zero() -> Self {
//         (0.0, 0.0).into()
//     }
// }

impl From<SpriteVector<f32>> for SpriteVectorFloat {
    fn from(value: SpriteVector<f32>) -> Self {
        (value.x, value.y).into()
    }
}

impl Mul<Coefficient> for SpriteVectorFloat {
    type Output = SpriteVectorFloat;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        let x = self.get_x() * rhs.0;
        let y = self.get_y() * rhs.0;
        (x,y).into()
    }
}

impl Mul<Coefficient> for SpriteVectorInt {
    type Output = SpriteVectorInt;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        let x = self.x * rhs.0 as i32;
        let y = self.y * rhs.0 as i32;
        (x,y).into()
    }
}

// impl VectorLike2d<f32> for SpriteVectorFloat {
//     fn get_x(self) -> f32 {
//         self.0.x
//     }

//     fn get_y(self) -> f32 {
//         self.0.y
//     }

//     fn dot(self, rhs: Self) -> f32 {
//         Vector::dot(self.0, rhs.0)
//     }

//     fn magnitude(self) -> f32 {
//         Vector::magnitude(self.0)        
//     }

//     fn distance(self, rhs: Self) -> f32 {
//         Vector::distance(self.0, rhs.0)
//     }
// }


// impl VectorLike2d<i32> for SpriteVectorInt {
//     fn get_x(self) -> i32 {
//         self.0.x
//     }

//     fn get_y(self) -> i32 {
//         self.0.y
//     }

//     fn dot(self, rhs: Self) -> i32 {
//         Vector::dot(self.0, rhs.0)
//     }

//     fn magnitude(self) -> f32 {
//         let m = self.iter()
//             .map(|n| {
//                 n * n
//             })
//             .sum::<i32>() as f32;
//         m.sqrt()
//     }

//     fn distance(self, rhs: Self) -> f32 {
//         let differences = self
//             .iter()
//             .zip(rhs.iter())
//             .map(|(a, b)| a - b);

//         let d = differences.map(|n| n * n).sum::<i32>() as f32;
//         d.sqrt()
//     }
// }



// fn calculate_reflection_vector_i32s(velocity: &SpriteVelocity, collision_normal: &SpriteVelocity) -> VelocityCalculationResults<i32> {
//     let mut calculations: VelocityCalculationResults<i32> = VelocityCalculationResults::<i32>::default();
//     calculations.initial = SpriteVectorInt::from(*velocity);
//     calculations.collision_normal = SpriteVectorInt::from(*collision_normal);
//     const SCALE: i32 = 1 << 10;
//     // let speed = velocity.magnitude();

//     // Ensure the normal is normalized (unit length)
//     calculations.normal = {
//         let mag = collision_normal.magnitude();
//         let mut x = calculations.collision_normal.x * SCALE;
//         let mut y = calculations.collision_normal.y * SCALE;
//         x = (x as f32 / mag).round() as i32;
//         y = (y as f32 / mag).round() as i32;
//         SpriteVectorInt::from((x,y))
//     };
//     // Calculate the component of the incoming velocity perpendicular to the collision surface
//     calculations.perpendicular = {
//         // velocity.perpendicular_velocity(&normal);
//         // let normal = collision_normal_normalized;
//         let dot = {
//             // (v.x * n.x * SCALE) + (v.y * n.x * SCALE) so we divide by scale to eliminate extra scaling
//             calculations.initial.dot(calculations.normal) / SCALE
//         };
//         calculations.normal.mul_c(dot)
//     };
//     // Calculate the component of the incoming velocity parallel to the collision surface
//     calculations.parallel = calculations.initial.mul_c(SCALE).parallel_velocity(&calculations.perpendicular);
//     // The reflected perpendicular velocity is reversed and scaled by the COR
//     calculations.reflected = calculations.perpendicular.neg();

//     // The reflected velocity is the sum of the reflected perpendicular and parallel components
//     calculations.reflected = calculations.reflected.add_assign(&calculations.parallel);

//     calculations.reflected.x /= SCALE;
//     calculations.reflected.y /= SCALE;
//     calculations
// }

// fn calculate_reflection_vector(velocity: &SpriteVelocity, collision_normal: &SpriteVelocity) -> SpriteVelocity {
//     let calc1: VelocityCalculationResults<i32> = Self::calculate_reflection_vector_i32s(velocity, collision_normal);
//     info!("calc1: {}", calc1);
//     let mut calc2: VelocityCalculationResults<f32> = VelocityCalculationResults::<f32>::default();
//     calc2.initial = SpriteVectorFloat::from(*velocity);
//     calc2.collision_normal = SpriteVectorFloat::from(*collision_normal);

//     // let speed = velocity.magnitude();
//     // Ensure the normal is normalized (unit length)
//     calc2.normal = calc2.collision_normal.normalize();

//     // Calculate the component of the incoming velocity perpendicular to the collision surface
//     calc2.perpendicular = calc2.initial.perpendicular_velocity(&calc2.normal);

//     // Calculate the component of the incoming velocity parallel to the collision surface
//     calc2.parallel = calc2.initial.parallel_velocity(&calc2.perpendicular);

//     // The reflected perpendicular velocity is reversed and scaled by the COR
//     calc2.reflected = calc2.perpendicular.neg();

//     // The reflected velocity is the sum of the reflected perpendicular and parallel components
//     calc2.reflected = calc2.reflected.add_assign(&calc2.parallel);

//     info!("calc2: {}", calc2);
//     // let speed2 = reflection.magnitude();
//     // if (speed - speed2).abs() >= 3.0*f32::EPSILON {
//     //     info!("speed changed by {}", (speed - speed2));
//     // }
//     // let ref_norm = reflection.normalize();
//     // ref_norm * Coefficient(speed)
//     calc2.reflected.into()
// }


pub trait VVelocity: Copy + Clone + PartialEq + defmt::Format + Add<Output = Self> + AddAssign + Neg<Output = Self> + Sub<Output = Self>
where 
    Self: From<Point> + From<(i32,i32)> + From<(f32,f32)> + Mul<Coefficient, Output = Self>, 
    // SpriteVector<f32>: From<Self>
{
    /// calculate the dot product of two vectors
    fn dot(self, rhs: Self) -> f32;

    /// Compute the magnitude of a vector
    fn magnitude(self) -> f32;

    // /// Compute the distance between two vectors
    // fn distance(self, rhs: Self) -> f32;

    #[inline]
    fn speed(&self) -> f32 {
        self.magnitude()
    }

    #[inline]
    fn direction(&self) -> Self {
        self.normalize()
    }

    fn zero() -> Self;

    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    /// try not to divide by zero 
    fn normalize(&self) -> Self;

    /// rotate the vector 90 degrees
    /// set x to negative y , and y to x
    fn rotate90(self) -> Self;
    // {
    //     let x = -self.get_y();
    //     let y = self.get_x();
    //     (x, y).into()
    // }


    /// Calculate the reflection off of the surface in the collision, given the surface_normal
    /// 
    fn calculate_reflection(&self, collision_normal: &Self) -> Self {
        let velocity = self;

        let speed = velocity.speed();
        let direction = velocity.direction();

        let surface_normal_speed = collision_normal.speed();
        let surface_normal_directon = collision_normal.direction();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        // this is the direction of the colision normal scaled by 
        // the dot product of self and the direction of the collision normal
        let v_perpendicular = {
            surface_normal_directon * Coefficient(velocity.dot(surface_normal_directon))
        };

        // Calculate the component of the incoming velocity parallel to the collision surface
        // this is simply self minus the perpendicular velocity
        let v_parallel = *velocity - v_perpendicular;

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular = -v_perpendicular;
        
        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        let reflected = reflected_perpendicular + v_parallel;

        {
            info!("initial_velocity: {}, s: {}, d: {}", velocity ,speed, direction);
            info!("surface_normal:   {}, s: {}, d: {}", collision_normal, surface_normal_speed, surface_normal_directon);
            info!("v_perp: {}, v_parallel: {}, reflected: {}", v_perpendicular, v_parallel, reflected);
            // let mut calculations = VelocityCalculationResults::<f32>::default();
            // calculations.initial = (*velocity).into();
            // calculations.collision_normal = (*collision_normal).into();
            // calculations.normal = surface_normal_directon.into();
            // calculations.perpendicular = v_perp.into();
            // calculations.parallel = parallel.into();
            // calculations.reflected = reflected.into();
            // info!("calculations: {}", calculations);
        }

        reflected
    }

}


impl VectorXY<f32> for SpriteVectorFloat {
    fn get_x(&self) -> f32 {
        self.x
    }

    fn get_y(&self) -> f32 {
        self.y
    }
}

impl VVelocity for SpriteVectorFloat {
    fn dot(self, rhs: Self) -> f32 {
        self.0.dot(rhs.0)
    }

    fn magnitude(self) -> f32 {
        self.0.magnitude()
    }

    fn zero() -> Self {
        Self(Vector2d::<f32>::from((0_f32,0_f32)))
    }

    fn normalize(&self) -> Self {
        let mag =  self.0.magnitude();
        if mag as i32 == 0 {
            VVelocity::zero()
        } else {
            let x = self.x / mag;
            let y = self.y / mag;
            (x, y).into()
        }
    }

    fn rotate90(self) -> Self {
        todo!()
    }
}
