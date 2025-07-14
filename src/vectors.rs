use core::ops::{Add, Deref, Div, Mul, Neg, Sub};
use core::fmt::Debug;
use embedded_graphics::prelude::Point;

use micromath::vector::{Component, Vector, Vector2d};
use num_traits::{self, AsPrimitive, Float, FromPrimitive, Inv, NumCast, Pow, ToPrimitive};

// use defmt::info;
// use thiserror_no_std::Error;

extern crate micromath as mm;
// extern crate nalgebra as na;
// use na::Vector2;

use crate::geometry::PointExt;

pub trait VecComp: 
    Component
    + ToPrimitive
    + FromPrimitive
    + core::ops::Neg<Output = Self>
{}

#[derive(Clone, Copy, Debug)]
pub struct GfxVector<C: VecComp>(pub Vector2d<C>);

impl<C: VecComp> Mul<C> for GfxVector<C> {
    type Output = Self;

    fn mul(self, rhs: C) -> Self::Output {
        Self(Vector2d { 
            x: self.x * rhs, 
            y: self.y * rhs, 
        })
    }
}

impl<C: VecComp> Mul<f32> for GfxVector<C> where C: Into<f32> + From<f32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let x: f32 = self.x.into();
        let y: f32 = self.y.into();
        Self(Vector2d { 
            x: (x * rhs).into(), 
            y: (y * rhs).into(), 
        })
    }
}

impl<C: VecComp> Mul<GfxVector<C>> for f32 where C: Into<f32> + From<f32> {
    type Output = GfxVector<C>;

    fn mul(self, rhs: GfxVector<C>) -> Self::Output {
        rhs.mul(self)
    }
}

impl<C: VecComp> Add<GfxVector<C>> for GfxVector<C> {
    type Output = Self;

    fn add(self, rhs: GfxVector<C>) -> Self::Output {
        Self(Vector2d { 
            x: self.x + rhs.x, 
            y: self.y + rhs.y
        })
    }
}

impl<C: VecComp> Sub<GfxVector<C>> for GfxVector<C> {
    type Output = Self;

    fn sub(self, rhs: GfxVector<C>) -> Self::Output {
        Self(Vector2d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        })
    }
}
impl<C: VecComp> Sub<GfxVector<C>> for &GfxVector<C> {
    type Output = GfxVector<C>;

    fn sub(self, rhs: GfxVector<C>) -> Self::Output {
        GfxVector(Vector2d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        })
    }
}

impl<C: VecComp> core::ops::DerefMut for GfxVector<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<C: VecComp> core::ops::Deref for GfxVector<C> {
    type Target = Vector2d<C>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<C: VecComp> core::ops::Neg for GfxVector<C> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(Vector2d { x: -self.x, y: -self.y })
    }
}

impl<C: VecComp> GfxVector<C> {
    fn magnitude(&self) -> C {
        let m = (self.x * self.x) + (self.y * self.y);
        let m: f32 = m.to_f32().unwrap().sqrt();
        let m: C = C::from_f32(m).unwrap();
        m
    }

    fn dot_product(&self, other: &Self) -> C {
        // (self.x * other.x) + (self.y * other.y)
        self.dot(*other.deref())
    }

    pub fn normalize(self) -> Self {
        let mag = self.magnitude();
        let mut norm = self.clone();
        norm.x = self.x / mag;
        norm.y = self.y / mag;
        norm
    }

    fn rotate90(self) -> Self {
        let x = -self.y;
        let y = self.x;
        Self(Vector2d { x, y })
    }

    pub fn calculate_reflection_vector(
        incoming_velocity: &Self,
        collision_normal: &Self,
        coefficient_of_restitution: f32,
    ) -> Self where C: From<f32> {
        // Ensure the normal is normalized (unit length)
        let normal = collision_normal.normalize();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        let perpendicular_velocity = normal * incoming_velocity.dot_product(&normal);

        // Calculate the component of the incoming velocity parallel to the collision surface
        let parallel_velocity = incoming_velocity - perpendicular_velocity;

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular_velocity = -perpendicular_velocity * coefficient_of_restitution.into();

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        reflected_perpendicular_velocity + parallel_velocity
    }    

}

impl<C: VecComp>  GfxVector<C> {
    pub fn new(x: C, y: C) -> Self {
        Self(Vector2d { x, y })
    }
}

impl<C: VecComp> From<Vector2d<C>> for GfxVector<C> {
    fn from(value: Vector2d<C>) -> Self {
        GfxVector(value)
    }
}

impl<C: VecComp> From<GfxVector<C>> for Point {
    fn from(value: GfxVector<C>) -> Self {
        let x= value.x.to_i32().unwrap();
        let y = value.y.to_i32().unwrap();
        Point { x, y }
    }
}

impl<C: VecComp + defmt::Format> defmt::Format for GfxVector<C> {
    fn format(&self, fmt: defmt::Formatter) {        
        defmt::write!(fmt, "{{ x: {}, y: {} }}", self.x, self.y)
    }
}



/// Let's define some stuff for handling vector manipulations for object collisions
pub trait VectorComponent:
    Copy
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Send
    + Sized
    + Sync
    // + NumOps
    + Pow<u8>
    + NumCast
    + Neg
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + ToPrimitive
    + FromPrimitive
    // + Neg<Output = Self>
{
}
impl VectorComponent for i8 {}
impl VectorComponent for i16 {}
impl VectorComponent for i32 {}
impl VectorComponent for f32 {}

// `i32: core::convert::From<f32>`
// `f32: core::convert::From<i32>`

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default, defmt::Format)]
pub struct SpriteVector<T: VectorComponent> {    
    pub x: T,
    pub y: T,
}

impl<T: VectorComponent + core::convert::From<i32>> From<Point> for SpriteVector<T> {
    fn from(value: Point) -> Self {
        SpriteVector {
            x: value.x.into(),
            y: value.y.into(),
        }
    }
}

impl<T: VectorComponent> From<SpriteVector<T>> for Point where i32: From<T> {
    fn from(value: SpriteVector<T>) -> Self {
        Point { x: value.x.into(), y: value.y.into() }
    }
}

impl<T: VectorComponent> Mul<SpriteVector<T>> for f32 {
    type Output = SpriteVector<T>;

    fn mul(self, rhs: SpriteVector<T>) -> Self::Output {
        let x = <f32 as num_traits::NumCast>::from(rhs.x).unwrap() * self;
        let y = <f32 as num_traits::NumCast>::from(rhs.y).unwrap() * self;
        SpriteVector {
            x: <T as num_traits::NumCast>::from(x).unwrap(),
            y: <T as num_traits::NumCast>::from(y).unwrap(),
        }
    }
}

impl<T: VectorComponent> Mul<T> for SpriteVector<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        SpriteVector {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: VectorComponent> Sub<SpriteVector<T>> for &SpriteVector<T> {
    type Output = SpriteVector<T>;

    fn sub(self, rhs: SpriteVector<T>) -> Self::Output {
        SpriteVector { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}
impl<T: VectorComponent> Sub<&SpriteVector<T>> for SpriteVector<T> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}
impl<T: VectorComponent> Sub for SpriteVector<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}
impl<T: VectorComponent> Add for SpriteVector<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}


trait NewTrait<T: VectorComponent + Neg<Output = T>> where SpriteVector<T>: Neg<Output = SpriteVector<T>> {
    fn new(x: T, y: T) -> Self;

    fn magnitude(&self) -> T;

    fn dot_product(&self, other: &Self) -> T;

    // fn distance_squared(&self, other: &Self) -> T {
    //     let delta_x = self.x - other.x;
    //     let delta_y = self.y - other.y;
    //     (delta_x * delta_x) + (delta_y * delta_y)
    // }

    // fn distance(&self, other: &Self) -> T {
    //     let d = self.distance_squared(other);

    //     let d = <f32 as num_traits::NumCast>::from(d).unwrap().sqrt();
    //     <T as num_traits::NumCast>::from(d).unwrap()
    // }
    
    fn normalize(self) -> Self;

    fn rotate90(self) -> Self;

    fn calculate_reflection_vector(
        incoming_velocity: &Self,
        collision_normal: &Self,
        coefficient_of_restitution: f32,
    ) -> Self;    

}

impl<T: VectorComponent + Neg<Output = T>> NewTrait<T> for SpriteVector<T> where SpriteVector<T>: Neg<Output = SpriteVector<T>> {
     fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

     fn magnitude(&self) -> T {
        let m: T = (self.x * self.x) + (self.y * self.y);
        let m = <f32 as num_traits::NumCast>::from(m).unwrap().sqrt();
        <T as num_traits::NumCast>::from(m).unwrap()
    }

    fn dot_product(&self, other: &Self) -> T {
        (self.x * other.x) + (self.y * other.y)
    }

    // fn distance_squared(&self, other: &Self) -> T {
    //     let delta_x = self.x - other.x;
    //     let delta_y = self.y - other.y;
    //     (delta_x * delta_x) + (delta_y * delta_y)
    // }

    // fn distance(&self, other: &Self) -> T {
    //     let d = self.distance_squared(other);

    //     let d = <f32 as num_traits::NumCast>::from(d).unwrap().sqrt();
    //     <T as num_traits::NumCast>::from(d).unwrap()
    // }
    
     fn normalize(self) -> Self {
        let mag = self.magnitude();
        let mut norm = self.clone();
        norm.x = self.x / mag;
        norm.y = self.y / mag;
        norm
    }

     fn rotate90(self) -> Self {
        let x = -self.y;
        let y = self.x;
        Self { x, y }
    }

     fn calculate_reflection_vector(
        incoming_velocity: &Self,
        collision_normal: &Self,
        coefficient_of_restitution: f32,
    ) -> Self {
        // Ensure the normal is normalized (unit length)
        let normal = collision_normal.normalize();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        let perpendicular_velocity = normal * incoming_velocity.dot_product(&normal);

        // Calculate the component of the incoming velocity parallel to the collision surface
        let parallel_velocity = incoming_velocity - perpendicular_velocity;

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular_velocity = coefficient_of_restitution * (-perpendicular_velocity);

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        reflected_perpendicular_velocity + parallel_velocity
    }    

}

pub trait VecNormalize {
    fn normalize(&self) -> Self;
}

impl<T> VecNormalize for Vector2d<T> 
where 
    T: micromath::vector::Component + core::convert::From<f32>,
    f32: core::convert::From<T>,
{
    fn normalize(&self) -> Self {
        let mag = self.magnitude();
        let x = <f32 as core::convert::From<T>>::from(self.x) / mag;
        let y = <f32 as core::convert::From<T>>::from(self.y) / mag;
        Vector2d { x: x.into(), y: y.into()}
    }
}

impl VecNormalize for Point {
    fn normalize(&self) -> Self {
        let mag = (self.x.pow(2) + self.y.pow(2)).isqrt();
        Self { x: self.x / mag, y: self.y / mag }
    }
}


fn normalize_vector2d_f32(a_vec: micromath::vector::Vector2d<f32>) -> micromath::vector::Vector2d<f32> {
    let mag = micromath::vector::Vector::magnitude(a_vec);
    micromath::vector::Vector2d { x: a_vec.x / mag, y: a_vec.y / mag }
}

fn calculate_reflection_vector<T>(
    incoming_velocity: &Vector2d<T>,
    collision_normal: &Vector2d<T>,
    coefficient_of_restitution: f32,
) -> Vector2d<T> 
where 
    Vector2d<T>: VecNormalize + Mul<f32, Output = Vector2d<T>> + Mul<T, Output = Vector2d<T>>,
    T: micromath::vector::Component + core::ops::Neg<Output = T>, 
    <T as Neg>::Output: Mul<f32>, 
    T: From<f32> + Mul<f32, Output = T>, 
{
    // Ensure the normal is normalized (unit length)
    // let mag = collision_normal.magnitude()
    let normal: Vector2d<T> = collision_normal.normalize();

    // Calculate the component of the incoming velocity perpendicular to the collision surface
    let perpendicular_velocity: Vector2d<T> = normal * incoming_velocity.dot(normal);
    let neg_x = -perpendicular_velocity.x;
    let neg_y = -perpendicular_velocity.y;
    let neg_perpendicular_velocity: Vector2d<T> = Vector2d { x: neg_x , y: neg_y };

    // Calculate the component of the incoming velocity parallel to the collision surface
    let parallel_velocity: Vector2d<T> = *incoming_velocity - perpendicular_velocity;

    // The reflected perpendicular velocity is reversed and scaled by the COR
    let reflected_perpendicular_velocity: Vector2d<T> = neg_perpendicular_velocity * coefficient_of_restitution;

    // The reflected velocity is the sum of the reflected perpendicular and parallel components
    reflected_perpendicular_velocity + parallel_velocity
}    

pub fn calculate_reflection_vector_p(
    incoming_velocity: &Point,
    collision_normal: &Point,
    coefficient_of_restitution: f32,
) -> Point {
    // Ensure the normal is normalized (unit length)
    // let mag = collision_normal.magnitude()
    let normal = collision_normal.normalize();

    // Calculate the component of the incoming velocity perpendicular to the collision surface
    let perpendicular_velocity = normal * incoming_velocity.dot_product(normal);
    let neg_x = -perpendicular_velocity.x;
    let neg_y = -perpendicular_velocity.y;
    let neg_perpendicular_velocity = Point { x: neg_x , y: neg_y };

    // Calculate the component of the incoming velocity parallel to the collision surface
    let parallel_velocity = *incoming_velocity - perpendicular_velocity;

    // The reflected perpendicular velocity is reversed and scaled by the COR
    let reflected_perpendicular_velocity = neg_perpendicular_velocity * coefficient_of_restitution as i32;

    // The reflected velocity is the sum of the reflected perpendicular and parallel components
    reflected_perpendicular_velocity + parallel_velocity
}    

