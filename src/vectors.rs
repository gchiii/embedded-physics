use core::ops::{Add, Deref, Mul, Neg, Sub};
use core::fmt::Debug;
use embedded_graphics::prelude::Point;

use micromath::vector::{Component, Vector, Vector2d};
use num_traits::{self, AsPrimitive, Float, FromPrimitive, Inv, NumCast, Pow, ToPrimitive};

// use defmt::info;
// use thiserror_no_std::Error;

extern crate micromath as mm;

pub trait VecComp: 
    Component
    + ToPrimitive
    + FromPrimitive
    + core::ops::Neg<Output = Self>
{}

impl VecComp for i32 {}
impl VecComp for f32 {}


#[derive(Clone, Copy, Debug)]
pub struct GfxVector<C: VecComp>(pub Vector2d<C>);


impl<C: VecComp + From<i32>> From<Point> for GfxVector<C> {
    fn from(value: Point) -> Self {
        GfxVector(Vector2d { 
            x: value.x.into(), 
            y: value.y.into() 
        })
    }
}

impl<C: VecComp + From<i32>> From<&Point> for GfxVector<C> {
    fn from(value: &Point) -> Self {
        GfxVector(Vector2d { 
            x: value.x.into(), 
            y: value.y.into() 
        })
    }
}

impl Mul<GfxVector<i32>> for i32 {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: GfxVector<i32>) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs.x * self, 
            y: rhs.y * self,
        })
    }
}

impl Mul<i32> for GfxVector<i32> {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: i32) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs * self.x, 
            y: rhs * self.y,
        })
    }
}

impl<'a> Mul<i32> for &'a GfxVector<i32> {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: i32) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs * self.x, 
            y: rhs * self.y,
        })
    }
}

impl Mul<GfxVector<f32>> for f32 {
    type Output = GfxVector<f32>;

    fn mul(self, rhs: GfxVector<f32>) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs.x * self, 
            y: rhs.y * self,
        })
    }
}

impl Mul<GfxVector<i32>> for f32 {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: GfxVector<i32>) -> Self::Output {
        let x = self * rhs.x as f32;
        let y = self * rhs.y as f32;
        GfxVector(Vector2d { x: x as i32, y: y as i32 })
    }
}

impl Mul<f32> for GfxVector<i32> {
    type Output = GfxVector<i32>;
    
    fn mul(self, rhs: f32) -> Self::Output {
        let x = rhs * self.x as f32;
        let y = rhs * self.y as f32;
        GfxVector(Vector2d { x: x as i32, y: y as i32 })
    }
}

impl<'a> Mul<&'a GfxVector<i32>> for f32 {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: &'a GfxVector<i32>) -> Self::Output {
        (*rhs).mul(self)
    }
}

impl<'a> Mul<f32> for &'a GfxVector<i32> {
    type Output = GfxVector<i32>;

    fn mul(self, rhs: f32) -> Self::Output {
        (*self).mul(rhs)
    }
}

impl<'a> Mul<f32> for &'a GfxVector<f32> {
    type Output = GfxVector<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs * self.x, 
            y: rhs * self.y,
        })
    }
}
impl<'a> Mul<&'a GfxVector<f32>> for f32 {
    type Output = GfxVector<f32>;

    fn mul(self, rhs: &'a GfxVector<f32>) -> Self::Output {
        GfxVector(Vector2d { 
            x: rhs.x * self, 
            y: rhs.y * self,
        })
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
impl<'a, C: VecComp> Sub<GfxVector<C>> for &'a GfxVector<C> {
    type Output = GfxVector<C>;

    fn sub(self, rhs: GfxVector<C>) -> Self::Output {
        GfxVector(Vector2d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        })
    }
}

impl<'a, C: VecComp> Sub<&'a GfxVector<C>> for GfxVector<C> {
    type Output = GfxVector<C>;

    fn sub(self, rhs: &'a GfxVector<C>) -> Self::Output {
        GfxVector(Vector2d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        })
    }
}

impl<'a, C: VecComp> Sub<&'a GfxVector<C>> for &'a GfxVector<C> {
    type Output = GfxVector<C>;

    fn sub(self, rhs: &'a GfxVector<C>) -> Self::Output {
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
impl<C: VecComp> From<(C,C)> for GfxVector<C> {
    fn from(value: (C,C)) -> Self {
        Self(value.into())
    }
}

pub trait VectorXY<C: VecComp> {
    fn get_x(&self) -> C;
    fn get_y(&self) -> C;
}

pub trait SquareRoot {
    fn calc_sqrt(&self) -> Self;
}

impl SquareRoot for i32 {
    fn calc_sqrt(&self) -> i32 {
        self.isqrt()
    }
}

impl SquareRoot for f32 {
    fn calc_sqrt(&self) -> f32 {
        self.sqrt()
    }
}

pub trait VectorTrait<C: VecComp + SquareRoot>: VectorXY<C>
where 
    Self: From<(C, C)> + Sized + Neg<Output = Self> + Sub<Output = Self> + Add<Output = Self> + for<'a> Sub<&'a Self, Output = Self>, 
    for<'a> &'a Self: Sub<&'a Self, Output = Self> + Mul<C, Output = Self>, 
    f32: Mul<Self, Output = Self> 
{
    /// get the magnitude of the vector in units C 
    fn magnitude(&self) -> C {
        let m = (self.get_x() * self.get_x()) + (self.get_y() * self.get_y());
        m.calc_sqrt()
        // let m: f32 = m.to_f32().unwrap().sqrt();
        // let m: C = C::from_f32(m).unwrap();
        // m
    }

    /// calculate the dot product of this vector and another one
    fn dot_product(&self, other: &Self) -> C {
        (self.get_x() * other.get_x()) + (self.get_y() * other.get_y())
    }

    /// Normalize the vector into a unit vector (keep direction, but magnitude is one)
    fn normalize(&self) -> Self {
        let mag = self.magnitude();        
        let x = self.get_x() / mag;
        let y = self.get_y() / mag;
        (x, y).into()
    }

    /// rotate the vector 90 degrees
    fn rotate90(self) -> Self {
        let x = -self.get_y();
        let y = self.get_x();
        (x, y).into()
    }

    /// Calculate the component of the incoming velocity perpendicular to the collision surface
    fn perpendicular_velocity(&self, collision_normal_normalized: &Self) -> Self {
        let normal = collision_normal_normalized;
        let dot = self.dot_product(normal);
        normal * dot
    }


    /// Calculate the component of the incoming velocity parallel to the collision surface
    fn parallel_velocity(&self, perpendicular_velocity: &Self) -> Self {
        self - perpendicular_velocity
    }
    
    fn calculate_reflection_vector(&self, collision_normal: &Self) -> Self {
        // Ensure the normal is normalized (unit length)
        let normal = collision_normal.normalize();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        let perpendicular_velocity = self.perpendicular_velocity(&normal);

        // Calculate the component of the incoming velocity parallel to the collision surface
        let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular_velocity = -perpendicular_velocity;

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        reflected_perpendicular_velocity + parallel_velocity
    }    

    fn calculate_reflection_vector_cor(&self, collision_normal: &Self, coefficient_of_restitution: f32) -> Self {
        // Ensure the normal is normalized (unit length)
        let normal = collision_normal.normalize();

        // Calculate the component of the incoming velocity perpendicular to the collision surface
        let perpendicular_velocity = self.perpendicular_velocity(&normal);

        // Calculate the component of the incoming velocity parallel to the collision surface
        let parallel_velocity = self.parallel_velocity(&perpendicular_velocity);

        // The reflected perpendicular velocity is reversed and scaled by the COR
        let reflected_perpendicular_velocity = coefficient_of_restitution * (-perpendicular_velocity);

        // The reflected velocity is the sum of the reflected perpendicular and parallel components
        reflected_perpendicular_velocity + parallel_velocity

    }

}

impl<C: VecComp> VectorXY<C> for GfxVector<C> {
    fn get_x(&self) -> C {
        self.x
    }

    fn get_y(&self) -> C {
        self.y
    }
}

impl VectorTrait<i32> for GfxVector<i32> {}
impl VectorTrait<f32> for GfxVector<f32> {}

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
    let incoming_velocity = GfxVector::<i32>::from(incoming_velocity);
    let collision_normal = GfxVector::<i32>::from(collision_normal);
    let reflection = GfxVector::calculate_reflection_vector_cor(&incoming_velocity, &collision_normal, coefficient_of_restitution);
    reflection.into()
}    

