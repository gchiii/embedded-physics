use embedded_graphics::{geometry::AnchorPoint, prelude::{Point, Size}, primitives::{Circle, Line, Polyline, Rectangle, Triangle}};
use num_traits::Float;


pub trait PointExt {
    fn magnitude(&self) -> i32;
    
    /// dot product of two points, self and other
    fn dot_product(&self, other: Point) -> i32;

    /// distance squared between two points
    fn distance_squared(&self, other: Point) -> i32;

    /// distance betwen points
    fn distance(&self, other: Point) -> i32;

    fn cross_product(&self, point: Point) -> i32;
}

impl PointExt for Point {
    fn cross_product(&self, point: Point) -> i32 {
        (self.x * point.y) - (self.y * point.x)
    }
    fn dot_product(&self, other: Point) -> i32 {
        (self.x * other.x) + (self.y * other.y)
    }

    fn distance_squared(&self, other: Point) -> i32 {
        (self.x - other.x).pow(2) + (self.y - other.y).pow(2)
    }

    fn distance(&self, other: Point) -> i32 {
        self.distance_squared(other).isqrt()
    }
    
    fn magnitude(&self) -> i32 {
        (self.x * self.x) + (self.y * self.y).isqrt()
    }
}

pub trait Area {
    fn area(&self) -> u32;
}

impl Area for Size {
    fn area(&self) -> u32 {
        self.height * self.width
    }
}

pub trait SurfaceDistance {
    /// distance from nearest surface to point
    fn distance(&self, point: impl Into<Point> + Copy) -> f32;
}

pub trait SurfaceNormal {
    /// compute the Vector Normal to the Surface between self and point as a Point
    fn surface_normal(&self, point: Point) -> Point;
}

pub trait ClosestEdge {
    fn closest_edge(&self, point: Point) -> Line;
}

pub trait ClosestPoint {
    fn closest_point(&self, point: Point) -> Point;
}

impl ClosestPoint for Line {
    fn closest_point(&self, point: Point) -> Point {
        let line = self;
        let line_vec = line.delta();
        let point_vec = point - line.start;
        let line_len_sq = line.start.distance_squared(line.end) as f32;

        if line_len_sq == 0.0 { 
            line.start
        } else {              
            let t = point_vec.dot_product(line_vec) as f32 / line_len_sq;
            let t = t.clamp(0.0, 1.0);

            Point {
                x: Float::round(line.start.x as f32 + t * line_vec.x as f32) as i32,
                y: Float::round(line.start.y as f32 + t * line_vec.y as f32) as i32,
            }
        }
    }
}

impl ClosestEdge for &[Point] {
    fn closest_edge(&self, point: Point) -> Line {
        let vertices = *self;
        if vertices.len() >= 2 {
            let mut closest1 = vertices[0];
            let mut closest2 = vertices[1];
    
            if closest1.distance_squared(point) > closest2.distance_squared(point) {
                core::mem::swap(&mut closest1, &mut closest2);
            }
    
            for &p in vertices.iter().skip(2) {
                let dist = p.distance_squared(point);
                if dist < closest1.distance_squared(point) {
                    closest2 = closest1;
                    closest1 = p;
                } else if dist < closest2.distance_squared(point) {
                    closest2 = p;
                }
            }
            Line::new(closest1, closest2)
        } else {
            Line::new(self[0], self[0])
        }
    }
}

impl SurfaceNormal for Line {
    fn surface_normal(&self, point: Point) -> Point {        
        let closest_point = self.closest_point(point);
        if closest_point == point {
            point
        } else {
            closest_point - point
        }
    }    
}

impl SurfaceDistance for Line {
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        let point: Point = point.into();
        let closest_point = self.closest_point(point);
        closest_point.distance(point) as f32
    }        
}

impl SurfaceNormal for Rectangle {
    fn surface_normal(&self, point: Point) -> Point {
        self.closest_edge(point).surface_normal(point)
    }
}

impl SurfaceDistance for Rectangle {    
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        self.closest_edge(point.into()).distance(point)
    }
}

impl ClosestEdge for Rectangle {
    fn closest_edge(&self, point: Point) -> Line {
        let vertices = [
            self.anchor_point(AnchorPoint::TopLeft),
            self.anchor_point(AnchorPoint::TopRight),
            self.anchor_point(AnchorPoint::BottomRight),
            self.anchor_point(AnchorPoint::BottomLeft)
        ];
        vertices.as_slice().closest_edge(point)
    }
}

impl ClosestEdge for Triangle {
    fn closest_edge(&self, point: Point) -> Line {
        self.vertices.as_slice().closest_edge(point)
    }
}

impl SurfaceNormal for Triangle {
    fn surface_normal(&self, point: Point) -> Point {
        self.closest_edge(point).surface_normal(point)
    }
    
}

impl SurfaceDistance for Triangle {
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        let point: Point = point.into();
        self.closest_edge(point).distance(point)
    }
}

impl<'a> ClosestEdge for Polyline<'a> {
    fn closest_edge(&self, point: Point) -> Line {
        self.vertices.closest_edge(point)
    }
}

impl<'a> SurfaceNormal for Polyline<'a> {
    fn surface_normal(&self, point: Point) -> Point {
        self.closest_edge(point).surface_normal(point)
    }
    
}

impl<'a> SurfaceDistance for Polyline<'a> {
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        let point: Point = point.into();
        self.closest_edge(point).distance(point)
    }
}

impl SurfaceNormal for Circle {
    // compute the surface normal between self and point
    fn surface_normal(&self, point: Point) -> Point {
        self.center() - point
    }
    
}

impl SurfaceDistance for Circle {
    fn distance(&self, point: impl Into<Point> + Copy) -> f32 {
        (self.center().distance(point.into()) as u32 - (self.diameter / 2)) as f32
    }
}

pub fn intersect_lines_i32(first_line: Line, other_line: Line) -> Option<Point> {   
    let a1 = first_line.end.y - first_line.start.y;
    let b1 = first_line.start.x - first_line.end.x;
    let c1 = (a1 * first_line.start.x) + (b1 * first_line.start.y);

    let a2 = other_line.end.y - other_line.start.y;
    let b2 = other_line.start.x - other_line.end.x;
    let c2 = (a2 * other_line.start.x) + (b2 * other_line.start.y);

    let delta = a1 * b2 - a2 * b1;

    if delta == 0 {
        return None;
    }

    let x = (b2 * c1 - b1 * c2) / delta;
    let y = (a1 * c2 - a2 * c1) / delta;
    Some(Point { x, y})
}

// fn intersect_lines(first_line: Line, other_line: Line) -> Option<Point> {        
//     let a1 = first_line.end.y - first_line.start.y;
//     let b1 = first_line.start.x - first_line.end.x;
//     let c1 = (a1 * first_line.start.x) + (b1 * first_line.start.y);

//     let a2 = other_line.end.y - other_line.start.y;
//     let b2 = other_line.start.x - other_line.end.x;
//     let c2 = (a2 * other_line.start.x) + (b2 * other_line.start.y);

//     let delta = a1 * b2 - a2 * b1;

//     if delta == 0 {
//         return None;
//     }

//     let x = F32((b2 * c1 - b1 * c2) as f32 / delta as f32);
//     let y = F32((a1 * c2 - a2 * c1) as f32 / delta as f32);
//     Some(Point {
//         x: x.round().0 as i32,
//         y: y.round().0 as i32,
//     })
// }

pub enum LineRelation {
    Parallel,
    Colinear,
    Intersect(Point),
    None,
}

pub fn intersection(l1: &Line, l2: &Line) -> LineRelation {
    let a = l1.start;
    let b = l1.end;
    let c = l2.start;
    let d = l2.end;

    let t_num = (c - a).cross_product(d - c);
    let t_den = (b - a).cross_product(d - c);

    let u_num = (a - c).cross_product(b - a);
    let u_den = (c - d).cross_product(b - a);

    let t = t_num as f32 / t_den as f32;
    let u = u_num as f32 / u_den as f32;

    if t_num == 0 && t_den == 0 && u_num == 0 && u_den == 0 {
        LineRelation::Colinear
    } else if t_den == 0 && u_den == 0 {
        LineRelation::Parallel
    } else if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        let p = b - a;
        let x = p.x as f32 * t;
        let y = p.y as f32 * t;
        let p = Point::new(x.round() as i32, y.round() as i32);
        LineRelation::Intersect(a + p)
    } else {
        LineRelation::None
    }
}


// // slope / intercept form of a line is y = mx + b
// fn calculat_slope_intercept_eq(p1: Point, p2: Point) {
//     let delta = p2 - p1;
//     let y_intercept = p1.y - ((delta.y * p1.x) / delta.x);
// }


#[cfg(feature = "nalgebra")]
fn calculate_standard_form(p1: Point, p2: Point) -> (i32, i32, i32) {
    let delta = p2 - p1;
    // slope is (delta.y / delta.x)
    // let y_intercept = p1.y - ((delta.y * p1.x) / delta.x);
    
    // coefficient of x
    let mut a = delta.y;
    // coefficient of y
    let mut b = delta.x;

    // since constant is y_intercept * delta.x, it simplifies to the following
    let mut c = p1.y - (delta.y * p1.x);

    if a.is_negative() {
        a *= -1;
        b *= -1;
        c *= -1;        
    }
    (a, b, c)
}

#[cfg(feature = "nalgebra")]
extern crate nalgebra as na;
#[cfg(feature = "nalgebra")]
pub struct LineEquationStdForm {
    /// x coefficient
    a: f32,
    /// y coefficient
    b: f32,
    /// constant
    c: f32,
}

#[cfg(feature = "nalgebra")]
impl LineEquationStdForm {
    pub fn from_points(p1: Point, p2: Point) -> Self {
        let (x_coefficient, y_coefficient, constant) = calculate_standard_form(p1, p2);
        Self { a: x_coefficient as f32, b: y_coefficient as f32, c: constant as f32}
    }
    
    // use num_complex::Complex;
    pub fn intersect(&self, other: &Self) -> Option<na::Matrix2x1<f32>> {
        // Create the coefficient matrix A
        let matrix_a = na::Matrix2::new(
            self.a, self.b,
            other.a, other.b,
        );

        // Create the constant vector B
        let vector_b = na::Vector2::new(self.c, other.c);

        // Solve for the intersection point (x, y)
        // This involves calculating the inverse of A and multiplying by B
        match matrix_a.try_inverse() {
            Some(inverse_a) => {
                let intersection_point = inverse_a * vector_b;
                // println!("Intersection point: {:?}", intersection_point);
                Some(intersection_point)
            }
            None => {
                // println!("Lines are parallel or coincident (no unique intersection).");
                None
            }
        }
    }
}
