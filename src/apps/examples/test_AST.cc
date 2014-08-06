

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#define PI 3.1415926535897932385
#define LO 0.0000000000
#define HI 4.0000000000
#define NUM_FUNC 2
#include <mra/mra.h>
#include <random>
#include <vector>

using namespace madness;

static const double R = 1.4;    // bond length
static const double L = 64.0*R; // box size
static const long k = 8;        // wavelet order
static const double thresh = 1e-6; // precision


static double sin_amp = 1.0;
static double cos_amp = 1.0;
static double sin_freq = 1.0;
static double cos_freq = 1.0;
static double sigma_x = 1.0;
static double sigma_y = 1.0;
static double sigma_z = 1.0;
static double center_x = 0.0;
static double center_y = 0.0;
static double center_z = 0.0;
static double gaussian_amp = 1.0;
static double sigma_sq_x = sigma_x*sigma_x;
static double sigma_sq_y = sigma_y*sigma_y;
static double sigma_sq_z = sigma_z*sigma_z;

static double random_function(const coord_3d& r) {
   const double x=r[0], y=r[1], z=r[2];

   const double dx = x-center_x;
   const double dy = y-center_y;
   const double dz = z-center_z;
   
   const double periodic_part = sin_amp * sin( sin_freq*(dx+dy+dz)) 
       + cos_amp * cos(cos_freq*(dx+dy+dz));
   
   const double x_comp = dx*dx/sigma_sq_x;
   const double y_comp = dy*dy/sigma_sq_y;
   const double z_comp = dz*dz/sigma_sq_z;
   
   const double gaussian_part = -gaussian_amp/exp(sqrt(x_comp+y_comp+z_comp));

   return gaussian_part*periodic_part;
}

static double get_rand()
{
    double r3 = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r3;
}

static void randomizer()
{
    sin_amp = get_rand();
    cos_amp = get_rand();
    sin_freq = get_rand();
    cos_freq = get_rand();
    sigma_x = get_rand();
    sigma_y = get_rand();
    sigma_z = get_rand();
    center_x = get_rand()*L/(2.0*HI);
    center_y = get_rand()*L/(2.0*HI);
    center_z = get_rand()*L/(2.0*HI);
    gaussian_amp = get_rand();
    sigma_sq_x = sigma_x*sigma_x;
    sigma_sq_y = sigma_y*sigma_y;
    sigma_sq_z = sigma_z*sigma_z;

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

    srand(time(NULL));

    Vector<Translation,3> l(0);
    l[0] = 62;
    l[1] = 60;
    l[2] = 62;	   
    FunctionImpl<double,3ul>::set_debug(Key<3ul>(4));
    
    real_function_3d farray[NUM_FUNC];// = new real_factory_3d[NUM_FUNC];
    AST_3D ast_func[NUM_FUNC];
    std::vector<AST_3D> ast_inter;

    double normr,start,end;

    for(int i = 0; i<NUM_FUNC; i++)
    {
	randomizer();
	farray[i] = real_factory_3d(world).f(random_function);	

	normr = farray[i].norm2();
	if (world.rank() == 0) 
	    print("Norm of function ",i," is ", normr);
	
	ast_func[i] = AST_3D(farray[i]);

    }

    for(int i =0; i<NUM_FUNC; i++)
    {
	for(int j = 0; j<=i ; j++){
	    AST_3D inter(1,farray[i],farray[j]);
	    ast_inter.push_back(inter);
	}	    

    }

    AST_3D final_AST(0,ast_inter);
    real_factory_3d result_factory(world);    
    world.gop.fence();

    std::cout<<"Starting Computation using AST"<<std::endl;    
    start = madness::wall_time();

    real_function_3d result(result_factory,final_AST);

    end = madness::wall_time();

    result.reconstruct();
    normr = result.norm2();
    
    if (world.rank() == 0) {
    	std::cout<<"Running time is "<<(end-start)<<"seconds"<<std::endl;	    
    	print("Norm of the result is ", normr);
    }

    world.gop.fence();
    
    std::cout<<std::endl<<"Starting Computation using original "<<std::endl;    
    start = madness::wall_time();

    real_function_3d inter[NUM_FUNC*(NUM_FUNC+1)/2];
    int count = 0;
    for(int i =0; i<NUM_FUNC; i++)
    {
	for(int j = 0; j<=i ; j++){
	    inter[count++] = farray[i]*farray[j];	    	    
	}	    

    }

    real_function_3d result_final = inter[0];
    for(int i =1; i<NUM_FUNC*(NUM_FUNC+1)/2;i++){
    	result_final = result_final + inter[i];
    }
    
    end = madness::wall_time();
    
    result_final.reconstruct();
    normr = result_final.norm2();
    
    if (world.rank() == 0) {
    	std::cout<<"Running time is "<<(end-start)<<"seconds"<<std::endl;	    
    	print("Norm of the result is ", normr);
    }

    world.gop.fence();
    
    
    





    //AST_3D a1(fa1);
    //AST_3D a2(fa2);
    //AST_3D a3(fa3);
    //AST_3D i1(1,a2,a3,a1);
    //AST_3D i(0,a3,i1);
    //
    //world.gop.fence();
    //std::cout<<"Starting"<<std::endl;
    //double start = madness::wall_time();
    //real_function_3d r1 = f1 * f2;
    //real_function_3d r2 = r1 * f3; 
    //real_function_3d r = f3 + r2;
    //world.gop.fence();
    //
    //double end = madness::wall_time();
    //
    //
    //r.reconstruct();
    ////spsi.print_tree();
    //normr = r.norm2();
    //if (world.rank() == 0) {
    //	std::cout<<"Running time is "<<(end-start)<<"seconds";	    
    //	print("              Norm is ", normr);
    //}
    //
    



    
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
