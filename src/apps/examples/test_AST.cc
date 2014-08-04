

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#define PI 3.1415926535897932385

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

static double VV(const coord_3d& r) {
    const double x=r[0], y=r[1], z=r[2];
    return -2.0/(sqrt(x*x+y*y+z*z+1e-8));
}


int main(int argc, char** argv) {
    typedef AST<double,3> AST_3D;
    typedef std::list<AST_3D> listAST;

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
//


    Vector<Translation,3> l(0);
    l[0] = 62;
    l[1] = 60;
    l[2] = 62;	   
    FunctionImpl<double,3ul>::set_debug(Key<3ul>(4));

    real_function_3d f1 = real_factory_3d(world).f(V);
    real_function_3d f2  = real_factory_3d(world).f(guess);
    real_function_3d f3 = real_factory_3d(world).f(VV);

    
    real_function_3d fa1 = real_factory_3d(world).f(V);    
    real_function_3d fa2  = real_factory_3d(world).f(guess);
    real_function_3d fa3  = real_factory_3d(world).f(VV);
    real_factory_3d result_factory(world);
    
    AST_3D a1(fa1);
    AST_3D a2(fa2);
    AST_3D a3(fa3);
    AST_3D i1(1,a2,a3,a1);
    AST_3D i(0,a1,i1);
    double normr;
    std::cout<<"Starting"<<std::endl;
    double start = madness::wall_time();
    real_function_3d r1 = f1 * f2;
    real_function_3d r2 = r1 * f3; 
    real_function_3d r = f1 + r2;
    world.gop.fence();

    double end = madness::wall_time();


    r.reconstruct();
    //spsi.print_tree();
    normr = r.norm2();
    if (world.rank() == 0) {
	std::cout<<"Running time is "<<(end-start)<<"seconds";	    
 	print("              Norm is ", normr);
    }
    std::cout<<"Starting AST"<<std::endl;
    start = madness::wall_time();
    real_function_3d result(result_factory,i);
    end = madness::wall_time();
    result.reconstruct();
    //result.print_tree();
    normr = result.norm2();
    
    if (world.rank() == 0) {
	std::cout<<"Running time is "<<(end-start)<<"seconds";	    
    	print("              Norm is ", normr);
    }




    
    //real_function_3d func3 = real_factory_3d(world).f(sinnx);

    
    ////a1._impl->print_tree();

    //AST_3D a3(func3);
    //
    
    //result.reconstruct();
    //normr = result.norm2();
    //result.reconstruct();
    //result.print_tree();
    //func1.print_tree();
//


    world.gop.fence();

    finalize();
    return 0;
}
