#include <medusa/Medusa.hpp>
#include <medusa/bits/domains/BasicRelax.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

/// Basic medusa example, we are solving 2D Poisson's equation on unit square
/// with Dirichlet boundary conditions.
/// http://e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation

using namespace mm; // NOLINT

int main() {

  auto continuous_boundary = [](Vec<double, 1> t) {
    double x, y;
    if (t(0) <= 4) { // Bottom boundary
      // t spans from 0 to 4
      x = t(0);
      y = -1.0 - 0.04 * std::sin(8 * M_PI * t(0) / 4);
    } else if (t(0) <= 6) { // Right boundary
      // t spans from 4 to 6, normalized to [0, 1] for calculation
      double normalizedT = (t(0) - 4) / 2;
      x = 4.0;
      y = -1.0 + normalizedT * 2; // Linearly interpolate y from -1 to 1
    } else if (t(0) <= 10) {      // Top boundary
      // t spans from 6 to 10, normalized to [0, 1] for calculation
      double normalizedT = (t(0) - 6) / 4;
      x = 4.0 - normalizedT * 4; // x goes from 4 to 0
      y = 1.0 - 0.04 * std::sin(8 * M_PI * normalizedT);
    } else { // Left boundary
      // t spans from 10 to 12, normalized to [0, 1] for calculation
      double normalizedT = (t(0) - 10) / 2;
      x = 0.0;
      y = 1.0 - normalizedT * 2; // Linearly interpolate y from 1 to -1
    }
    return Vec2d(x, y);
  };

  auto der_continuous_boundary = [](Vec<double, 1> t) {
    double der_x = 0, der_y = 0;
    if (t(0) <= 4) { // Bottom boundary
      der_x = 1.0;
      der_y = -0.02 * M_PI * std::cos(2 * M_PI * t(0) / 4);
    } else if (t(0) <= 6) { // Right boundary
      der_x = 0;
      der_y = 1.0;
    } else if (t(0) <= 10) { // Top boundary
      der_x = -1.0;
      der_y = -0.02 * M_PI * std::cos(2 * M_PI * (t(0) - 6) / 4);
    } else { // Left boundary
      der_x = 0;
      der_y = -1.0;
    }
    Eigen::Matrix<double, 2, 1> jm;
    jm << der_x, der_y;
    return jm;
  };

  double dx = 0.05;
  BoxShape<Vec1d> param_bs(Vec<double, 1>{0.0}, Vec<double, 1>{12.0});
  UnknownShape<Vec2d> shape;
  DomainDiscretization<Vec2d> edge_domain(shape);
  // DomainDiscretization<Vec2d> domain =
  // shape.discretizeBoundaryWithStep(dx/5);

  // Gradient function for domain fill, modify as needed
  auto gradient_h = [](Vec2d p) {
    double h_0 = 0.005;
    double h_m = 0.03 - h_0;
    return (0.5 * h_m * (p(0) + p(1) + 3.0) + h_0) / 5.0;
  };

  // Fill the domain boundary using the continuous parametric function
  GeneralSurfaceFill<Vec2d, Vec1d> gsf;
  edge_domain.fill(gsf, param_bs, continuous_boundary, der_continuous_boundary,
                   dx / 10);
  // Construct the contains function.
  KDTree<Vec2d> contains_tree;
  edge_domain.makeDiscreteContainsStructure(contains_tree);
  auto contains_function = [&](const Vec2d p) {
    return edge_domain.discreteContains(p, contains_tree);
  };

  // Fill the interior
  DomainDiscretization<Vec2d> domain(shape);
  GeneralFill<Vec2d> gf;
  domain.fill(gsf, param_bs, continuous_boundary, der_continuous_boundary, dx);
  KDTreeMutable<Vec2d> tree;
  // domain.fill(gf, dx); // Fill the domain considering the gradient function
  gf(domain, dx, tree, contains_function);
  // Assign boundary types
  // Loop through all boundary points to assign BCs for top and bottom
  for (size_t i = 0; i < domain.boundary().size(); i++) {
    int index = domain.boundary()[i];
    Vec2d position =
        domain.positions()[index]; // Position of the boundary point

    // Liberal check for top and bottom boundaries
    if (position[1] >
        0) { // Any point above the horizontal midline is considered top
      domain.types()[index] = -4;
    } else { // Any point below the horizontal midline is considered bottom
      domain.types()[index] = -3;
    }
  }
  // Second pass for left and right boundaries
  // Second pass for left and right boundaries, avoiding reassignment at corners
  for (size_t i = 0; i < domain.boundary().size(); i++) {
    int index = domain.boundary()[i];
    Vec2d position =
        domain.positions()[index]; // Position of the boundary point

    // Reduced tolerance for determining closeness to the corner
    double cornerTolerance =
        1e-10; // Adjust based on observed behavior and computational needs

    // Check if the point is close to the corners, if so, skip reassignment
    bool isNearTopCorner = (std::abs(position[1] - 1.0) < cornerTolerance);
    bool isNearBottomCorner = (std::abs(position[1] + 1.0) < cornerTolerance);
    if ((std::abs(position[0] - 0.0) < cornerTolerance &&
         (isNearTopCorner || isNearBottomCorner)) ||
        (std::abs(position[0] - 4.0) < cornerTolerance &&
         (isNearTopCorner || isNearBottomCorner))) {
      // Skip this point, it's near a corner and should retain its initial
      // top/bottom BC assignment
      continue;
    }

    // Proceed with left/right boundary checks and reassignments
    if (std::abs(position[0] - 0.0) <
        cornerTolerance) { // Left boundary with the reduced tolerance
      domain.types()[index] = -1;
    } else if (std::abs(position[0] - 4.0) <
               cornerTolerance) { // Right boundary with the reduced tolerance
      domain.types()[index] = -2;
    }
    // Top and bottom assignments are preserved for corner points by the
    // continue statement
  }

  // Create cyl
  BallShape<Vec2d> cyl({0.6, 0.0}, 0.25);
  DomainDiscretization<Vec2d> cyl_dom = cyl.discretizeBoundaryWithStep(dx);
  for (size_t i = 0; i < cyl_dom.boundary().size(); i++) {
    cyl_dom.types()[cyl_dom.boundary()[i]] =
        cyl_dom.types()[cyl_dom.boundary()[i]] - 4;
  }

  // Final domain
  domain -= cyl_dom;

  BasicRelax relax;
  double min_step = 0.5 * dx;
  // relax.iterations(1000).initialHeat(1).finalHeat(0).boundaryProjectionThreshold(0.75).projectionType(BasicRelax::PROJECT_IN_DIRECTION).numNeighbours(6);
  relax.iterations(50)
      .initialHeat(1)
      .numNeighbours(4)
      .finalHeat(0.1)
      .projectionType(BasicRelax::PROJECT_IN_DIRECTION)
      .boundaryProjectionThreshold(0.35);
  // relax.iterations(10)
  //     .initialHeat(3)
  //     .finalHeat(0.0)
  //     .projectionType(BasicRelax::DO_NOT_PROJECT)
  //     .numNeighbours(4);
  // relax.iterations(100).initialHeat(5).boundaryProjectionThreshold(1.0).projectionType(BasicRelax::PROJECT_IN_DIRECTION).numNeighbours(3);
  // relax.iterations(20).initialHeat(1).numNeighbours(3).boundaryProjectionThreshold(1.5).projectionType(BasicRelax::PROJECT_IN_DIRECTION);

  relax(domain, min_step);

  // Find support for the nodes
  // int N = domain.size ();
  domain.findSupport(
      FindClosest(9)); // The support for each node is the closest 9 nodes

  //// Override to print points
  // Extract interior nodes, boundary nodes, and normals from the domain.
  Range<int> interior = domain.interior();
  Range<int> boundary = domain.boundary();
  Range<Vec2d> normals = domain.normals();
  // Extract the positions of the nodes.
  Range<Vec2d> positions = domain.positions();
  // Exract boundary map
  Range<int> boundary_map = domain.bmap();
  // Extract types of the nodes.
  Range<int> types = domain.types();
  // Print interior to file
  std::ofstream myfile;
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_interior.txt");
  for (int i = 0; i < interior.size(); i++) {
    myfile << interior[i] << "\n";
  }
  myfile.close();
  // Print boundary to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_boundary.txt");
  for (int i = 0; i < boundary.size(); i++) {
    myfile << boundary[i] << "\n";
  }
  myfile.close();
  // Print normals to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_normals.txt");
  for (int i = 0; i < normals.size(); i++) {
    myfile << normals[i][0] << " , " << normals[i][1] << "\n";
  }
  myfile.close();
  // Print positions to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_positions.txt");
  for (int i = 0; i < positions.size(); i++) {
    myfile << positions[i][0] << " , " << positions[i][1] << "\n";
  }
  myfile.close();
  // Print boundary map to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_boundary_map.txt");
  for (int i = 0; i < boundary_map.size(); i++) {
    myfile << boundary_map[i] << "\n";
  }
  myfile.close();
  // Print types to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "wavy_cyl_0_05_types.txt");
  for (int i = 0; i < types.size(); i++) {
    myfile << types[i] << "\n";
  }
  myfile.close();
  return 0;
}
