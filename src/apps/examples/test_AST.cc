

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <mra/mra.h>

using namespace madness;

static const double R = 1.4;    // bond length
static const double L = 64.0*R; // box size
static const long k = 8;        // wavelet order
static const double thresh = 1e-6; // precision

static double guess(const coord_3d& r) {
    const double x=r[0], y=r[1], z=r[2];
    return (exp(-sqrt(x*x+y*y+(z-R/2)*(z-R/2)+1e-8))+
            exp(-sqrt(x*x+y*y+(z+R/2)*(z+R/2)+1e-8)));
}

static double V(const coord_3d& r) {
    const double x=r[0], y=r[1], z=r[2];
    return -1.0/sqrt(x*x+y*y+(z-R/2)*(z-R/2)+1e-8)+
           -1.0/sqrt(x*x+y*y+(z+R/2)*(z+R/2)+1e-8);
}


int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);

    startup(world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<3>::set_defaults(world);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L/2, L/2);

    real_function_3d Vnuc = real_factory_3d(world).f(V);
    real_function_3d psi  = real_factory_3d(world).f(guess);
    real_function_3d Vpsi = (Vnuc*psi);
    real_function_3d spsi = Vpsi+psi;
    double norm = spsi.norm2();
    spsi.compress();
    double normc = spsi.norm2();
    spsi.reconstruct();
    double normr = spsi.norm2();

    if (world.rank() == 0) {
	print("                    Norm is ", norm, normc, normr);
    }

    world.gop.fence();

    finalize();
    return 0;
}
