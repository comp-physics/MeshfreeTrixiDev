#include <medusa/Medusa_fwd.hpp>
#include <medusa/bits/domains/BasicRelax.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

/// Basic medusa example, we are solving 2D Poisson's equation on unit square
/// with Dirichlet boundary conditions.
/// http://e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation

using namespace mm; // NOLINT

int main() {
  // Geometry setup for Mach 3 flow around a cylinder in 2D with Compressible
  // Euler Equations Create the domain and discretize it
  // double dx = 0.00625;
  // double dx = 0.0125;
  // double dx = 0.010;
  // double dx = 0.005;
  double dx = 0.00125;
  BoxShape<Vec2d> box_whole({0.0, -1.0}, {4.0, 1.0});
  BallShape<Vec2d> cyl({0.6, 0.0}, 0.25);
  // BoxShape<Vec2d> box_whole({0.0, 0.0}, {2.0, 0.5});
  // BallShape<Vec2d> cyl({0.3, 0.25}, 0.125 / 2);
  // DomainDiscretization<Vec2d> box_whole_dom =
  // box_whole.discretizeWithStep(dx);
  DomainDiscretization<Vec2d> box_whole_dom =
      box_whole.discretizeBoundaryWithStep(dx);
  GeneralFill<Vec2d> fill_engine;
  fill_engine(box_whole_dom, dx);
  DomainDiscretization<Vec2d> cyl_dom = cyl.discretizeBoundaryWithStep(dx);
  // Shift boundary types for step
  // std::cout << cyl_dom.boundary().size() << std::endl;
  for (size_t i = 0; i < cyl_dom.boundary().size(); i++) {
    cyl_dom.types()[cyl_dom.boundary()[i]] =
        cyl_dom.types()[cyl_dom.boundary()[i]] - 4;
  }
  // cyl_dom.types()[cyl_dom.boundary()] = -5;
  box_whole_dom -= cyl_dom;
  auto domain = box_whole_dom;

  // Oversample the domain.
  // DomainDiscretization<Vec2d> oversampled_domain =
  // box.discretizeBoundaryWithStep(dx / 5);

  // Construct the contains function.
  // KDTree<Vec2d> contains_tree;
  // oversampled_domain.makeDiscreteContainsStructure(contains_tree);
  // auto contains_function = [&](const Vec2d p)
  // {
  //     return oversampled_domain.discreteContains(p, contains_tree);
  // };

  // Fill the boundary normally.
  // DomainDiscretization<Vec2d> domain = box.discretizeBoundaryWithStep(dx);

  // Fill the interior with the contains function.
  // KDTreeMutable<Vec2d> tree;
  // GeneralFill<Vec2d> gf;
  // gf(domain, dx, tree, contains_function);

  BasicRelax relax;
  // relax.iterations(20).initialHeat(5).numNeighbours(4).projectionType(BasicRelax::DO_NOT_PROJECT);
  // relax.iterations(20).initialHeat(5).numNeighbours(3).projectionType(BasicRelax::DO_NOT_PROJECT);
  // relax.iterations(1000).initialHeat(5).finalHeat(0).numNeighbours(3).projectionType(BasicRelax::DO_NOT_PROJECT);
  // relax.iterations(100).initialHeat(5).numNeighbours(4).projectionType(BasicRelax::DO_NOT_PROJECT);
  // relax.iterations(100).initialHeat(4).numNeighbours(4).projectionType(BasicRelax::DO_NOT_PROJECT);
  double min_step = 0.5 * dx;
  // relax(domain, min_step);
  // V2
  // BasicRelax relax;
  // relax.iterations(1000).initialHeat(1).finalHeat(0).boundaryProjectionThreshold(0.75).projectionType(BasicRelax::PROJECT_IN_DIRECTION).numNeighbours(6);
  // relax.iterations(250)
  //     .initialHeat(5)
  //     .finalHeat(0.0)
  //     .boundaryProjectionThreshold(1.0)
  //     .projectionType(BasicRelax::PROJECT_IN_DIRECTION)
  //     .numNeighbours(3);
  relax.iterations(20)
      .initialHeat(3)
      .finalHeat(0.0)
      .boundaryProjectionThreshold(1.0)
      .projectionType(BasicRelax::PROJECT_IN_DIRECTION)
      .numNeighbours(4);
  // relax.iterations(100).initialHeat(5).boundaryProjectionThreshold(1.0).projectionType(BasicRelax::PROJECT_IN_DIRECTION).numNeighbours(3);
  // relax.iterations(20).initialHeat(1).numNeighbours(3).boundaryProjectionThreshold(1.5).projectionType(BasicRelax::PROJECT_IN_DIRECTION);

  relax(domain, min_step);

  // Find support for the nodes
  int N = domain.size();
  domain.findSupport(
      FindClosest(9)); // the support for each node is the closest 9 nodes

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
              "cyl_0_00625_interior.txt");
  for (int i = 0; i < interior.size(); i++) {
    myfile << interior[i] << "\n";
  }
  myfile.close();
  // Print boundary to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "cyl_0_00125_boundary.txt");
  for (int i = 0; i < boundary.size(); i++) {
    myfile << boundary[i] << "\n";
  }
  myfile.close();
  // Print normals to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "cyl_0_00125_normals.txt");
  for (int i = 0; i < normals.size(); i++) {
    myfile << normals[i][0] << " , " << normals[i][1] << "\n";
  }
  myfile.close();
  // Print positions to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "cyl_0_00125_positions.txt");
  for (int i = 0; i < positions.size(); i++) {
    myfile << positions[i][0] << " , " << positions[i][1] << "\n";
  }
  myfile.close();
  // Print boundary map to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "cyl_0_00125_boundary_map.txt");
  for (int i = 0; i < boundary_map.size(); i++) {
    myfile << boundary_map[i] << "\n";
  }
  myfile.close();
  // Print types to file
  myfile.open("/Users/jarias9/Documents/MeshfreeTrixiDev/medusa_point_clouds/"
              "cyl_0_00125_types.txt");
  for (int i = 0; i < types.size(); i++) {
    myfile << types[i] << "\n";
  }
  myfile.close();
}
