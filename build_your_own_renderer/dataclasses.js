class Ray {
  constructor(origin, direction) {
    this.origin = origin;
    this.direction = direction;
  }
}

class Vector3D {
  constructor(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  add(other) {
    if (other instanceof Vector3D) {
      return new Vector3D(this.x + other.x, this.y + other.y, this.z + other.z);
    } else if (typeof other === 'number') {
      return new Vector3D(this.x + other, this.y + other, this.z + other);
    }

    throw TypeError('Add not implemented for type');
  }

  subtract(other) {
    if (other instanceof Vector3D) {
      return new Vector3D(this.x - other.x, this.y - other.y, this.z - other.z);
    } else if (typeof other === 'number') {
      return new Vector3D(this.x - other, this.y - other, this.z - other);
    }

    throw TypeError('Add not implemented for type');
  }

  dot(other) {
    if (!(other instanceof Vector3D)) {
      throw TypeError('Type must be vector');
    }

    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  norm() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  normalize() {
    const norm = this.norm();
    const vec = new Vector3D(this.x, this.y, this.z);
    return vec.scale(1 / norm);
  }

  scale(other) {
    if (typeof other === 'number') {
      return new Vector3D(this.x * other, this.y * other, this.z * other);
    }

    throw TypeError('Scale not implemented for type');
  }

  static lerp(x1, x2, alpha) {
    return x1.scale(1 - alpha).add(x2.scale(alpha));
  }
}

class Sphere {
  constructor(center, radius, material) {
    this.center = center;
    this.radius = radius;
    this.material = material;
  }

  checkCollisionWithRay(ray) {
    if (!(ray instanceof Ray)) {
      throw TypeError('Type must be ray');
    }

    const d = ray.direction;
    const c_dash = ray.origin.subtract(this.center);

    const a = Math.pow(d.norm(), 2);
    const b = 2 * c_dash.dot(d);
    const c = Math.pow(c_dash.norm(), 2) - Math.pow(this.radius, 2);

    const D = b * b - 4 * a * c;

    if (D < 0) {
      return [false, null];
    }

    const t1 = (-b + Math.sqrt(D)) / (2 * a);
    const t2 = (-b - Math.sqrt(D)) / (2 * a);

    // both before image plane
    if (t1 < 0 && t2 < 0) {
      return [false, null];
    }
    // one before one after
    else if (t1 * t2 < 0) {
      return [true, Math.max(t1, 2)];
    }
    // both after image plane
    return [true, Math.min(t1, t2)];
  }
}

class Material {
  constructor(k_a, k_d, k_s, alpha, k_r) {
    this.k_a = k_a; // color
    this.k_d = k_d; // color
    this.k_s = k_s; // color
    this.k_r = k_r; // color
    this.alpha = alpha;
  }
}

class Light {
  constructor(origin, i_d, i_s) {
    this.origin = origin;
    this.i_d = i_d; // color
    this.i_s = i_s; // color
  }
}

class Color {
  constructor(r, g, b) {
    this.r = r;
    this.g = g;
    this.b = b;
  }

  scale(other) {
    if (other instanceof Color) {
      return new Color(this.r * other.r, this.g * other.g, this.b * other.b);
    } else if (typeof other === 'number') {
      return new Color(this.r * other, this.g * other, this.b * other);
    }

    throw TypeError();
  }
  add(other) {
    return new Color(this.r + other.r, this.g + other.g, this.b + other.b);
  }
  clamp() {
    const clamp = (x) => {
      if (x < 0) {
        return 0;
      } else if (x > 1) {
        return 1;
      }
      return x;
    };

    return new Color(clamp(this.r), clamp(this.g), clamp(this.b));
  }
}
