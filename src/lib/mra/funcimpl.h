/*
 This file is part of MADNESS.

 Copyright (C) 2007,2010 Oak Ridge National Laboratory

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

 For more information please contact:

 Robert J. Harrison
 Oak Ridge National Laboratory
 One Bethel Valley Road
 P.O. Box 2008, MS-6367

 email: harrisonrj@ornl.gov
 tel:   865-241-3937
 fax:   865-572-0680


 $Id$
 */

#ifndef MADNESS_MRA_FUNCIMPL_H__INCLUDED
#define MADNESS_MRA_FUNCIMPL_H__INCLUDED
#define NOT_OPERATOR 999
#define ADD_OPERATOR 0
#define MULTIPLY_OPERATOR 1
#define RECONSTRUCT_OPERATOR 2
#define DIFF_OPERATOR 3
#define DEBUG 0

/// \file funcimpl.h
/// \brief Provides FunctionCommonData, FunctionImpl and FunctionFactory

#include <iostream>
#include <world/world.h>
#include <world/print.h>
#include <world/scopedptr.h>
#include <world/typestuff.h>
#include <misc/misc.h>
#include <tensor/tensor.h>
#include <cstdarg>
#include <vector>
#include <mra/function_common_data.h>
#include <mra/indexit.h>
#include <mra/key.h>
#include <mra/funcdefaults.h>
#include <mra/function_factory.h>

namespace madness {
    template<typename T, std::size_t NDIM>
    class FunctionImpl;

    template<typename T, std::size_t NDIM>
    class AST;

    template<typename T, std::size_t NDIM>
    class FunctionNode;

    template<typename T, std::size_t NDIM>
    class Function;

    template<typename T, std::size_t NDIM>
    class FunctionFactory;

}

namespace madness {


    /// A simple process map
    template<typename keyT>
    class SimplePmap : public WorldDCPmapInterface<keyT> {
    private:
        const int nproc;
        const ProcessID me;

    public:
        SimplePmap(World& world) : nproc(world.nproc()), me(world.rank()) 
        { }

        ProcessID owner(const keyT& key) const {
            if (key.level() == 0)
                return 0;
            else
                return key.hash() % nproc;
        }
    };

    /////////////////////////////////////////////////////////////////////////////////////////	
    /* ---------------------------Composing Operations using AST -------------------------*/
    ////////////////////////////////////////////////////////////////////////////////////////

    template<typename T, std::size_t NDIM> 
	class AST {
	
	friend class FunctionImpl<T,NDIM>; 
    public:
	typedef AST<T,NDIM> astT;
	typedef Tensor <T> coeffT;
	typedef Key<NDIM> keyT;
	typedef std::pair<keyT,coeffT> kcT;
	typedef std::list <AST <T,NDIM> > listT;
	typedef std::vector <AST <T,NDIM> > vecT;
	typedef FunctionImpl<T,NDIM> implT;
	typedef std::shared_ptr<implT> spimplT;

    private:

	
    public:
	spimplT _impl;
	kcT _key_coeff;
	vecT _operands;
	int _operator;
	bool _is_ready, _has_coefficients;

       
    AST():
	_impl(NULL),
	    _operator(NOT_OPERATOR),
	    _is_ready(false),
	    _has_coefficients(false)
	    {
	    }

    

	//Constructors
    AST(const Function<T,NDIM> &function):
	_impl(function.impl),	    
	    _operator(NOT_OPERATOR),
	    _is_ready(false),
	    _has_coefficients(false){
	    //std::cout<<"Function Impl Set"<<std::endl;
	    //_impl->print_tree();
	}
	

    AST(spimplT &impl):
	_impl(impl),
	    _operator(NOT_OPERATOR),
	    _is_ready(false),
	    _has_coefficients(false){
	    
	}
	
    AST(kcT& keyCoeff):
	    _impl(NULL),
		_key_coeff(keyCoeff),	
		_operator(NOT_OPERATOR),
		_is_ready(false),	    
		_has_coefficients(true)
		{
		}
	    
    AST(int op, vecT &operands):
	_impl(NULL),
	    _operands(operands),
	    _operator(op),
	    _is_ready(false),
	    _has_coefficients(false)
	    {
	    }

    AST(int op, astT a1, astT a2):    
	_impl(NULL),
	    _operator(op),    
	    _is_ready(false),
	    _has_coefficients(false)
	
	    {
		vecT operands;
		operands.push_back(a1);
		operands.push_back(a2);
		_operands = operands;
		    
	    }

    AST(int op, astT a1, astT a2, astT a3):
	_impl(NULL),
	    _operator(op),    
	    _is_ready(false),
	    _has_coefficients(false)
	    {
		vecT operands;
		operands.push_back(a1);
		operands.push_back(a2);
		operands.push_back(a3);
		_operands = operands;
		    
	    }
    AST(int op, astT a1, astT a2, astT a3, astT a4):_operator(op),    
	    _is_ready(false),
	    _has_coefficients(false),
	    _impl(NULL)
	    {
		vecT operands;
		operands.push_back(a1);
		operands.push_back(a2);
		operands.push_back(a3);
		operands.push_back(a4);
		_operands = operands;
		    
	    }
	
	//Copy Constructor
    AST(const AST& other):
	_impl(other._impl),
	_key_coeff(other._key_coeff),
	    _operands(other._operands),
	    _operator(other._operator),		
	    _is_ready(other._is_ready),
	    _has_coefficients(other._has_coefficients)

	    {
	    	    
	    } 

	bool is_operator() const
	{
	    return !(_operator==NOT_OPERATOR);
	}

	bool has_coeffs() const
	{
	    return _has_coefficients;
	}
	int get_operator() const
	{
	    return _operator;
	}

	implT* get_impl() const{
	    return _impl.get();
	}
	
	vecT  get_operand_list() const
	{
	    return _operands;
	}

        kcT get_key_coeff_pair() const
	{
	    return _key_coeff;
	}
	
	void set_ready(bool is_ready) 
	{
	    _is_ready = is_ready;
	}

	void set_has_coeff(bool has_coeff)
	{
	    _has_coefficients = has_coeff;
	}

	void set_key_coeff(kcT &keyCoeff)
	{
	    _key_coeff= keyCoeff;
	}

	void set_function(std::shared_ptr<FunctionImpl<T,NDIM>> function)
	{
	    _impl = function;
	}  

    };

    namespace archive {
        /// Serialize an AST
        template <class Archive, typename T, std::size_t NDIM>
	    struct ArchiveStoreImpl< Archive, AST<T,NDIM> > {
		static void store(const Archive& s, const AST<T,NDIM>& t) {
		    s & t.has_coeffs() & t.is_operator();
		    if(t.has_coeffs())
			s & t.get_key_coeff_pair();
		    else if(t.get_operator() == NOT_OPERATOR)
			
			s & t.get_impl();
		    else
			s & t.get_operator() & t.get_operand_list();
		}
        };
	
        /// Deserialize an AST 
        template <class Archive, typename T, std::size_t NDIM>
	    struct ArchiveLoadImpl< Archive, AST<T,NDIM> > {
            	typedef Tensor <T> coeffT;
		typedef Key<NDIM> keyT;
		typedef std::pair<keyT,coeffT> kcT;
		typedef std::vector <AST <T,NDIM> > vecT;
		
		
		static void load(const Archive& s, AST<T,NDIM>& t) {
		    
		    bool hasCoeff = false, isOperator = false;
		    s & hasCoeff & isOperator;

		    if(hasCoeff)
		    {
			kcT keyCoeff;
			s & keyCoeff;
			t = AST<T,NDIM>(keyCoeff);	

		    }
		    else if(!isOperator)
		    {
			typedef FunctionImpl<T,NDIM> implT;
			implT* impl = NULL;
			s & impl;
			std::shared_ptr<implT> simpl(impl);
			t = AST<T,NDIM>(simpl);
			
		    }
		    else
		    {
		    
			vecT operand_list;
			int op = 0;
			s & op & operand_list;
			t = AST<T,NDIM>(op, operand_list);
		    }
		
		}
        };




    }
	
    /////////////////////////////////////////////////////////////////////////////////////////
    /*---------------------------------------Ends Here-------------------------------------*/
    /////////////////////////////////////////////////////////////////////////////////////////


    /// FunctionNode holds the coefficients, etc., at each node of the 2^NDIM-tree
    template<typename T, std::size_t NDIM>
    class FunctionNode {
    public:
    	typedef Tensor<T> coeffT;
    	typedef Tensor<T> tensorT;
    private:
        // Should compile OK with these volatile but there should
        // be no need to set as volatile since the container internally
        // stores the entire entry as volatile

        coeffT _coeffs; ///< The coefficients, if any
        double _norm_tree; ///< After norm_tree will contain norm of coefficients summed up tree
        bool _has_children; ///< True if there are children
        coeffT buffer; ///< The coefficients, if any

    public:

        typedef WorldContainer<Key<NDIM> , FunctionNode<T, NDIM> > dcT; ///< Type of container holding the nodes
        /// Default constructor makes node without coeff or children
        FunctionNode() :
                _coeffs(), _norm_tree(1e300), _has_children(false) {
        }

        /// Constructor from given coefficients with optional children

        /// Note that only a shallow copy of the coeff are taken so
        /// you should pass in a deep copy if you want the node to
        /// take ownership.
        explicit
        FunctionNode(const coeffT& coeff, bool has_children = false) :
                _coeffs(coeff), _norm_tree(1e300), _has_children(has_children) {
        }

        explicit
        FunctionNode(const coeffT& coeff, double norm_tree, bool has_children) :
            _coeffs(coeff), _norm_tree(norm_tree), _has_children(has_children) {
        }

        FunctionNode(const FunctionNode<T, NDIM>& other) {
            *this = other;
        }

        FunctionNode<T, NDIM>&
        operator=(const FunctionNode<T, NDIM>& other) {
            if (this != &other) {
                coeff() = copy(other.coeff());
                _norm_tree = other._norm_tree;
                _has_children = other._has_children;
            }
            return *this;
        }

        /// Copy with possible type conversion of coefficients, copying all other state

        /// Choose to not overload copy and type conversion operators
        /// so there are no automatic type conversions.
        template<typename Q>
        FunctionNode<Q, NDIM>
        convert() const {
            return FunctionNode<Q, NDIM> (copy(coeff()), _has_children);
        }

        /// Returns true if there are coefficients in this node
        bool
        has_coeff() const {
        	return _coeffs.has_data();
        }

        bool exists() const {return this->has_data();}

        /// Returns true if this node has children
        bool
        has_children() const {
            return _has_children;
        }

        /// Returns true if this does not have children
        bool
        is_leaf() const {
            return !_has_children;
        }

        /// Returns true if this node is invalid (no coeffs and no children)
        bool
        is_invalid() const {
            return !(has_coeff() || has_children());
        }

        /// Returns a non-const reference to the tensor containing the coeffs

        /// Returns an empty tensor if there are no coefficients.
        coeffT&
        coeff() {
            MADNESS_ASSERT(_coeffs.ndim() == -1 || (_coeffs.dim(0) <= 2
                                                    * MAXK && _coeffs.dim(0) >= 0));
            return const_cast<coeffT&>(_coeffs);
        }

        /// Returns a const reference to the tensor containing the coeffs

        /// Returns an empty tensor if there are no coefficeints.
        const coeffT&
        coeff() const {
            return const_cast<const coeffT&>(_coeffs);
        }

        /// Returns the number of coefficients in this node
        size_t size() const {
        	return _coeffs.size();
        }

    public:

        /// Sets \c has_children attribute to value of \c flag.
        Void
        set_has_children(bool flag) {
            _has_children = flag;
            return None;
        }

        /// Sets \c has_children attribute to true recurring up to ensure connected
        Void
        set_has_children_recursive(const typename FunctionNode<T,NDIM>::dcT& c,const Key<NDIM>& key) {
            //madness::print("   set_chi_recu: ", key, *this);
            PROFILE_MEMBER_FUNC(FunctionNode);
            if (!(has_children() || has_coeff() || key.level()==0)) {
                // If node already knows it has children or it has
                // coefficients then it must already be connected to
                // its parent.  If not, the node was probably just
                // created for this operation and must be connected to
                // its parent.
                Key<NDIM> parent = key.parent();
                const_cast<dcT&>(c).task(parent, &FunctionNode<T,NDIM>::set_has_children_recursive, c, parent, TaskAttributes::hipri());
                //madness::print("   set_chi_recu: forwarding",key,parent);
            }
            _has_children = true;
            return None;
        }

        /// Sets \c has_children attribute to value of \c !flag
        void set_is_leaf(bool flag) {
            _has_children = !flag;
        }

        /// Takes a \em shallow copy of the coeff --- same as \c this->coeff()=coeff
        void set_coeff(const coeffT& coeffs) {
            coeff() = coeffs;
            if ((_coeffs.has_data()) and ((_coeffs.dim(0) < 0) || (_coeffs.dim(0)>2*MAXK))) {
                print("set_coeff: may have a problem");
                print("set_coeff: coeff.dim[0] =", coeffs.dim(0), ", 2* MAXK =", 2*MAXK);
            }
            MADNESS_ASSERT(coeffs.dim(0)<=2*MAXK && coeffs.dim(0)>=0);
        }

        /// Clears the coefficients (has_coeff() will subsequently return false)
        void clear_coeff() {
            coeff()=coeffT();
        }

        /// Scale the coefficients of this node
        template <typename Q>
        void scale(Q a) {
        	_coeffs.scale(a);
        }

        /// Sets the value of norm_tree
        Void set_norm_tree(double norm_tree) {
            _norm_tree = norm_tree;
            return None;
        }

        /// Gets the value of norm_tree
        double get_norm_tree() const {
            return _norm_tree;
        }


        /// General bi-linear operation --- this = this*alpha + other*beta

        /// This/other may not have coefficients.  Has_children will be
        /// true in the result if either this/other have children.
        template <typename Q, typename R>
        Void gaxpy_inplace(const T& alpha, const FunctionNode<Q,NDIM>& other, const R& beta) {
            PROFILE_MEMBER_FUNC(FuncNode);
            if (other.has_children())
                _has_children = true;
            if (has_coeff()) {
                if (other.has_coeff()) {
                    coeff().gaxpy(alpha,other.coeff(),beta);
                }
                else {
                    coeff().scale(alpha);
                }
            }
            else if (other.has_coeff()) {
                coeff() = other.coeff()*beta; //? Is this the correct type conversion?
            }
            return None;
        }

        template <typename Archive>
        void serialize(Archive& ar) {
            ar & coeff() & _has_children & _norm_tree;
        }

    };

    template <typename T, std::size_t NDIM>
    std::ostream& operator<<(std::ostream& s, const FunctionNode<T,NDIM>& node) {
        s << "(has_coeff=" << node.has_coeff() << ", has_children=" << node.has_children() << ", norm=";
        double norm = node.has_coeff() ? node.coeff().normf() : 0.0;
        if (norm < 1e-12)
            norm = 0.0;
        double nt = node.get_norm_tree();
        if (nt == 1e300) nt = 0.0;
        s << norm << ", norm_tree=" << nt << ")";
        return s;
    }

    

    /// FunctionImpl holds all Function state to facilitate shallow copy semantics

    /// Since Function assignment and copy constructors are shallow it
    /// greatly simplifies maintaining consistent state to have all
    /// (permanent) state encapsulated in a single class.  The state
    /// is shared between instances using a shared_ptr<FunctionImpl>.
    ///
    /// The FunctionImpl inherits all of the functionality of WorldContainer
    /// (to store the coefficients) and WorldObject<WorldContainer> (used
    /// for RMI and for its unqiue id).
    ///
    /// The class methods are public to avoid painful multiple friend template
    /// declarations for Function and FunctionImpl ... but this trust should not be
    /// abused ... NOTHING except FunctionImpl methods should mess with FunctionImplData.
    /// The LB stuff might have to be an exception.
    template <typename T, std::size_t NDIM>
    class FunctionImpl : public WorldObject< FunctionImpl<T,NDIM> > {
	template<typename Q,std::size_t NDIM1> friend class AST;

    private:
        typedef WorldObject< FunctionImpl<T,NDIM> > woT; ///< Base class world object type

    public:
        typedef FunctionImpl<T,NDIM> implT; ///< Type of this class (implementation)
	typedef std::list <AST <T,NDIM> > listT;
        typedef std::shared_ptr< FunctionImpl<T,NDIM> > pimplT; ///< pointer to this class
        typedef Tensor<T> tensorT; ///< Type of tensor for anything but to hold coeffs
        typedef Vector<Translation,NDIM> tranT; ///< Type of array holding translation
        typedef Key<NDIM> keyT; ///< Type of key
        typedef FunctionNode<T,NDIM> nodeT; ///< Type of node
        typedef Tensor<T> coeffT; ///< Type of tensor used to hold coeffs
        typedef WorldContainer<keyT,nodeT> dcT; ///< Type of container holding the coefficients
        typedef std::pair<const keyT,nodeT> datumT; ///< Type of entry in container
        typedef Vector<double,NDIM> coordT; ///< Type of vector holding coordinates
	typedef AST<T,NDIM> astT;
	typedef std::vector<astT> vecT;

        //template <atypename Q, int D> friend class Function;
        template <typename Q, std::size_t D> friend class FunctionImpl;

        World& world;

    private:
        int k; ///< Wavelet order
        double thresh; ///< Screening threshold
        int initial_level; ///< Initial level for refinement
        int max_refine_level; ///< Do not refine below this level
        int truncate_mode; ///< 0=default=(|d|<thresh), 1=(|d|<thresh/2^n), 1=(|d|<thresh/4^n);
        bool autorefine; ///< If true, autorefine where appropriate
        bool truncate_on_project; ///< If true projection inserts at level n-1 not n
        bool nonstandard; ///< If true, compress keeps scaling coeff

        const FunctionCommonData<T,NDIM>& cdata;

        std::shared_ptr< FunctionFunctorInterface<T,NDIM> > functor;

        bool on_demand; ///< does this function have an additional functor?
        bool compressed; ///< Compression status
        bool redundant; ///< If true, function keeps sum coefficients on all levels

        dcT coeffs; ///< The coefficients

        // Disable the default copy constructor
        FunctionImpl(const FunctionImpl<T,NDIM>& p);

    public:

	static keyT debugKey;

        /// Initialize function impl from data in factory
        FunctionImpl(const FunctionFactory<T,NDIM>& factory)
                : WorldObject<implT>(factory._world)
                , world(factory._world)
                , k(factory._k)
                , thresh(factory._thresh)
                , initial_level(factory._initial_level)
                , max_refine_level(factory._max_refine_level)
                , truncate_mode(factory._truncate_mode)
                , autorefine(factory._autorefine)
                , truncate_on_project(factory._truncate_on_project)
                , nonstandard(false)
                , cdata(FunctionCommonData<T,NDIM>::get(k))
                , functor(factory.get_functor())
                , on_demand(factory._is_on_demand)
                , compressed(false)
                , redundant(false)
                , coeffs(world,factory._pmap,false)
                //, bc(factory._bc)
            {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            // !!! Ensure that all local state is correctly formed
            // before invoking process_pending for the coeffs and
            // for this.  Otherwise, there is a race condition.
            MADNESS_ASSERT(k>0 && k<=MAXK);

            bool empty = (factory._empty or is_on_demand());
            bool do_refine = factory._refine;

            if (do_refine)
                initial_level = std::max(0,initial_level - 1);


		if (empty) { // Do not set any coefficients at all

		    // additional functors are only evaluated on-demand
		} else if (functor) { // Project function and optionally refine

		    insert_zero_down_to_initial_level(cdata.key0);

		    typename dcT::const_iterator end = coeffs.end();

		    for (typename dcT::const_iterator it=coeffs.begin(); it!=end; ++it) {

			if (it->second.is_leaf())
			    woT::task(coeffs.owner(it->first), &implT::project_refine_op, it->first, do_refine,
				      functor->special_points());
		    }
		}

		else { // Set as if a zero function

		    initial_level = 1;
		    insert_zero_down_to_initial_level(keyT(0));
		}

            coeffs.process_pending();
            this->process_pending();

            if (factory._fence && functor)
                world.gop.fence();
        }

        /// Copy constructor

        /// Allocates a \em new function in preparation for a deep copy
        ///
        /// By default takes pmap from other but can also specify a different pmap.
        /// Does \em not copy the coefficients ... creates an empty container.
        template <typename Q>
        FunctionImpl(const FunctionImpl<Q,NDIM>& other,
                     const std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >& pmap,
                     bool dozero)
                : WorldObject<implT>(other.world)
                , world(other.world)
                , k(other.k)
                , thresh(other.thresh)
                , initial_level(other.initial_level)
                , max_refine_level(other.max_refine_level)
                , truncate_mode(other.truncate_mode)
                , autorefine(other.autorefine)
                , truncate_on_project(other.truncate_on_project)
                , nonstandard(other.nonstandard)
                , cdata(FunctionCommonData<T,NDIM>::get(k))
                , functor()
                , on_demand(false)	// since functor() is an default ctor
                , compressed(other.compressed)
                , redundant(other.redundant)
                , coeffs(world, pmap ? pmap : other.coeffs.get_pmap())
                //, bc(other.bc)
        {
            if (dozero) {
                initial_level = 1;
                insert_zero_down_to_initial_level(cdata.key0);
            }
            coeffs.process_pending();
            this->process_pending();
        }

	static void set_debug(keyT key)
	{
	    debugKey = key;
	}

        virtual ~FunctionImpl() { }

        const std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >& get_pmap() const {
            return coeffs.get_pmap();
        }

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/*--------------------------------------- AST Tree Traversal --------------------------------------*/
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	coeffT get_coeff_from_parent(astT parent_ast, keyT child_key)
	{
	    keyT parent_key = parent_ast._key_coeff.first;
	    coeffT parent_coeff = parent_ast._key_coeff.second;
	    coeffT child_coeff = copy(parent_to_child(parent_coeff, parent_key, child_key));
	    return child_coeff;
	}

	astT add_ready_operands(vecT ready_operands_list, keyT& key)
	{
	    
	    coeffT result_coeff = get_coeff_from_parent(ready_operands_list[0],key);
	    
	    for(int i =1; i<ready_operands_list.size(); i++)
	    {
		coeffT operand = get_coeff_from_parent(ready_operands_list[i],key);
		result_coeff += operand;
	    }
	    
	    std::pair<keyT,coeffT> key_coeff(key,result_coeff);
	    astT result_ast(key_coeff);
	    return result_ast;

	}

	astT multiply_ready_operands(vecT ready_operands_list, keyT& key)
	{
	    int num_ready_operands = ready_operands_list.size();
	    if (num_ready_operands > 2)
	    {
		if(DEBUG && key == debugKey)
		    madness::print("Entering loop");
		
		//double product_norm = 1.0;
		//for(int i =0; i<ready_operands_list.size(); i++)
		//{
		//    product_norm *= ready_operands_list[i]._key_coeff.second.normf();
		//}

		    
		vecT evaluated_operand_list;
		for(int i =0; i<ready_operands_list.size()-1; i+=2)
		{
		    
		    
		    PROFILE_MEMBER_FUNC(FunctionImpl);


		    //madness::print("do_mul: r", rkey, rcoeff.size());
		    coeffT rcube = fcube_for_mul(key, ready_operands_list[i]._key_coeff.first, ready_operands_list[i]._key_coeff.second);
		    
		    //madness::print("do_mul: l", key, left.size());
		    coeffT lcube = fcube_for_mul(key, ready_operands_list[i+1]._key_coeff.first, ready_operands_list[i+1]._key_coeff.second);

		    coeffT tcube(cdata.vk,false);
		    TERNARY_OPTIMIZED_ITERATOR(T, tcube, T, lcube, T, rcube, *_p0 = *_p1 * *_p2;);
		    double scale = pow(0.5,0.5*NDIM*key.level())*sqrt(FunctionDefaults<NDIM>::get_cell_volume());
		    tcube = transform(tcube,cdata.quad_phiw).scale(scale);
		    
		    std::pair<keyT,coeffT> key_coeff(key,tcube);

		    astT result_ast(key_coeff);		    		    
		    evaluated_operand_list.push_back(result_ast);
		    
		}

		if(DEBUG && key == debugKey)
		    madness::print("adding the odd coeff");


		if (num_ready_operands % 2 !=0)
		    evaluated_operand_list.push_back(ready_operands_list[num_ready_operands-1]);

		astT result(MULTIPLY_OPERATOR,evaluated_operand_list);
		return result;

	    }else{
		PROFILE_MEMBER_FUNC(FunctionImpl);
		    //madness::print("do_mul: r", rkey, rcoeff.size());
		    coeffT rcube = fcube_for_mul(key, ready_operands_list[0]._key_coeff.first, ready_operands_list[0]._key_coeff.second);
		    
		    //madness::print("do_mul: l", key, left.size());
		    coeffT lcube = fcube_for_mul(key, ready_operands_list[1]._key_coeff.first, ready_operands_list[1]._key_coeff.second);

		    coeffT tcube(cdata.vk,false);
		    TERNARY_OPTIMIZED_ITERATOR(T, tcube, T, lcube, T, rcube, *_p0 = *_p1 * *_p2;);
		    double scale = pow(0.5,0.5*NDIM*key.level())*sqrt(FunctionDefaults<NDIM>::get_cell_volume());
		    tcube = transform(tcube,cdata.quad_phiw).scale(scale);
		    
		    std::pair<keyT,coeffT> key_coeff(key,tcube);

		    astT result_ast(key_coeff);		    		    
		    return result_ast;

	    }

	}

	//not implemented yet
	astT binary_op(int op, vecT ready_operands_list, keyT& key)
	{

	    if(ready_operands_list.size() == 1)
		return ready_operands_list[0];

	    if(DEBUG && key == debugKey)
		std::cout<<"About to Compute "<<std::endl;

	    if (op == ADD_OPERATOR)
		return add_ready_operands(ready_operands_list, key);

	    if (op == MULTIPLY_OPERATOR)
		return multiply_ready_operands(ready_operands_list, key);

	    //this should not happen
	    astT temp;
	    return temp;
	}

	/*exp represents a function node that may or may not have coefficients.
	  If coefficients are already present, then no need for further evaluation.
	  If coefficients are not present then check to see if coefficients are available at the key*/
	astT soft_evaluation(astT& exp, keyT& key)
	{
            typedef typename implT::dcT::const_iterator iterT;

	    if(exp._has_coefficients)
		return exp;

	    else{

		std::shared_ptr<implT> func = exp._impl;
		
		iterT it = func->coeffs.find(key).get();		
		
		if (it->second.has_coeff()){
		    //std::cout<<key<<std::endl;

		    coeffT temp = copy(it->second.coeff());
		    exp._key_coeff  = std::pair<keyT,coeffT>(key,temp);
		    exp._has_coefficients = true;

		    }
		
		return exp;		 
		}
	}


	//if the coefficients are available, checks if further refinement is necessary
	//based on the operation
	bool is_coeff_ready(int op, astT exp, keyT& key)
	{
	    if (!exp._has_coefficients) return false;

	    if (op == ADD_OPERATOR && key >= exp._key_coeff.first) return true;

	    if (op == MULTIPLY_OPERATOR && key >= exp._key_coeff.first) return true;
	    return false;

	}

	astT evaluate_AST(astT &exp, keyT& key)
	{
	    vecT not_ready_operands_list;
	    vecT ready_operands_list;
	    astT result_ast;

	    //binary operation
	    if (exp.is_operator() && exp._operands.size() > 1)
	    {
		if(DEBUG && key == debugKey)
		    std::cout<<key<<std::endl;
		int op = exp._operator;
		vecT* operands = &exp._operands;
	
		//evaluate each of the operands
		for(typename vecT::iterator it = operands->begin(); it != operands->end(); it++)
		{		
		    astT evaluated_operand = evaluate_AST(*it,key);
		
		    //if the evaluated operand has coefficients check if they are ready to be consumed
		    //put them in the ready list if they are
		    if(evaluated_operand._has_coefficients && is_coeff_ready(op,evaluated_operand,key))
		    {
			if(DEBUG && key == debugKey)
			    std::cout<<"OK operand ready "<<std::endl;

			ready_operands_list.push_back(evaluated_operand);	     
		    }
		    else
			not_ready_operands_list.push_back(evaluated_operand);

		
		}


		//if there more at least two operands ready then apply the operation on them
		astT result;
		int num_ready_operands = ready_operands_list.size();

		if(num_ready_operands > 0)
		    result = binary_op(op,ready_operands_list, key);

		if(not_ready_operands_list.size() >0)
		{		     
		    if(num_ready_operands > 0)	
			not_ready_operands_list.push_back(result);
		    
		    exp._operands = not_ready_operands_list;
		    return exp;
		}

		if(DEBUG && key == debugKey){
		    std::cout<<"Computed Coeff Norm is :"<<result._key_coeff.second.normf()<<std::endl;
		    std::cout<<"Computed and returning "<<std::endl;
		}
		
		return result;
	    }

	    //unary operation
	    else if (exp.is_operator())
	    {
		//do unary operation
		return exp;
	    }
	    else
	    {		
		return soft_evaluation(exp,key);

	    }
	}

	void traverse_tree(astT &exp, keyT key = keyT(0))
	{
	
	    if (world.rank() != coeffs.owner(key)) return;
	    //if(DEBUG)
		//std::cout<<key<<std::endl;
	    

	    astT evaluated_exp = evaluate_AST(exp,key);
	    if(DEBUG && key == debugKey)
	    {
		std::cout<<"Evaluated ";
		if( evaluated_exp._has_coefficients)
			std::cout<<"and Has Coefficients "<<std::endl;

	    }
	    if (evaluated_exp._has_coefficients)
	    {		
		if(DEBUG && key == debugKey){
		    std::cout<<"Replacing"<<std::endl;
		    std::cout<<"Result has Coeff "<<evaluated_exp._key_coeff.second.has_data()<<std::endl;
		}
		coeffs.replace(key,nodeT(evaluated_exp._key_coeff.second,false));
	    }
	    else 
	    {
		coeffs.replace(key, nodeT(coeffT(),true)); // Interior node

		for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {

                    const keyT& child = kit.key();
		    astT exp_copy = astT(evaluated_exp);
		    woT::task(coeffs.owner(child), &implT:: traverse_tree, exp_copy, child);
                }
	    }	    		

	}

	/*
	////////////////////////////////////////// END of AST stuff ////////////////////////////////////////
	*/

	
        /// Returns true if the function is compressed.
        bool is_compressed() const {
            return compressed;
        }

        /// Returns true if the function is redundant.
        bool is_redundant() const {
            return redundant;
        }

        bool is_nonstandard() const {return nonstandard;}

        void set_functor(const std::shared_ptr<FunctionFunctorInterface<T,NDIM> > functor1) {
        	this->on_demand=true;
        	functor=functor1;
        }

        std::shared_ptr<FunctionFunctorInterface<T,NDIM> > get_functor() {
        	MADNESS_ASSERT(this->functor);
        	return functor;
        }

        std::shared_ptr<FunctionFunctorInterface<T,NDIM> > get_functor() const {
            MADNESS_ASSERT(this->functor);
            return functor;
        }

        void unset_functor() {
        	this->on_demand=false;
        	functor.reset();
        }

        bool& is_on_demand() {return on_demand;};
        const bool& is_on_demand() const {return on_demand;};

        double get_thresh() const {return thresh;}

        void set_thresh(double value) {thresh = value;}

        bool get_autorefine() const {return autorefine;}

        void set_autorefine(bool value) {autorefine = value;}

        int get_k() const {return k;}

        const dcT& get_coeffs() const {return coeffs;}

        dcT& get_coeffs() {return coeffs;}

        const FunctionCommonData<T,NDIM>& get_cdata() const {return cdata;}

        /// Initialize nodes to zero function at initial_level of refinement.

        /// Works for either basis.  No communication.
        void insert_zero_down_to_initial_level(const keyT& key);

        /// Evaluate function at quadrature points in the specified box
        void fcube(const keyT& key, const FunctionFunctorInterface<T,NDIM>& f, const Tensor<double>& qx, tensorT& fval) const;
        void fcube(const keyT& key,  T (*f)(const coordT&), const Tensor<double>& qx, tensorT& fval) const;

        const keyT& key0() const {
            return cdata.key0;
        }

        void print_tree(Level maxlevel = 10000) const;

        void do_print_tree(const keyT& key, Level maxlevel) const;

        /// Compute by projection the scaling function coeffs in specified box
        tensorT project(const keyT& key) const;

        /// Returns the truncation threshold according to truncate_method

        /// here is our handwaving argument:
        /// this threshold will give each FunctionNodean error of less than tol. The
        /// total error can then be as high as sqrt(#nodes) * tol. Therefore in order
        /// to account for higher dimensions: divide tol by about the root of number
        /// of siblings (2^NDIM) that have a large error when we refine along a deep
        /// branch of the tree.
        double truncate_tol(double tol, const keyT& key) const {
            const static double fac=1.0/std::pow(2,NDIM*0.5);
            tol*=fac;

            // RJH ... introduced max level here to avoid runaway
            // refinement due to truncation threshold going down to
            // intrinsic numerical error
            const int MAXLEVEL1 = 20; // 0.5**20 ~= 1e-6
            const int MAXLEVEL2 = 10; // 0.25**10 ~= 1e-6

            if (truncate_mode == 0) {
                return tol;
            }
            else if (truncate_mode == 1) {
                double L = FunctionDefaults<NDIM>::get_cell_min_width();
                return tol*std::min(1.0,pow(0.5,double(std::min(key.level(),MAXLEVEL1)))*L);
            }
            else if (truncate_mode == 2) {
                double L = FunctionDefaults<NDIM>::get_cell_min_width();
                return tol*std::min(1.0,pow(0.25,double(std::min(key.level(),MAXLEVEL2)))*L*L);
            }
            else {
                MADNESS_EXCEPTION("truncate_mode invalid",truncate_mode);
            }
        }

        /// Returns patch referring to coeffs of child in parent box
        std::vector<Slice> child_patch(const keyT& child) const {
            std::vector<Slice> s(NDIM);
            const Vector<Translation,NDIM>& l = child.translation();
            for (std::size_t i=0; i<NDIM; ++i)
                s[i] = cdata.s[l[i]&1]; // Lowest bit of translation
            return s;
        }

        /// Projection with optional refinement
        Void project_refine_op(const keyT& key, bool do_refine,
                const std::vector<Vector<double,NDIM> >& specialpts);

        /// Compute the Legendre scaling functions for multiplication

        /// Evaluate parent polyn at quadrature points of a child.  The prefactor of
        /// 2^n/2 is included.  The tensor must be preallocated as phi(k,npt).
        /// Refer to the implementation notes for more info.
        void phi_for_mul(Level np, Translation lp, Level nc, Translation lc, Tensor<double>& phi) const;

        /// Returns the box at level n that contains the given point in simulation coordinates
        Key<NDIM> simpt2key(const coordT& pt, Level n) const {
            Vector<Translation,NDIM> l;
            double twon = std::pow(2.0, double(n));
            for (std::size_t i=0; i<NDIM; ++i) {
                l[i] = Translation(twon*pt[i]);
            }
            return Key<NDIM>(n,l);
        }

        template <typename Q>
        Tensor<Q> coeffs2values(const keyT& key, const Tensor<Q>& coeff) const {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            double scale = pow(2.0,0.5*NDIM*key.level())/sqrt(FunctionDefaults<NDIM>::get_cell_volume());
            return transform(coeff,cdata.quad_phit).scale(scale);
        }

        template <typename Q>
        Tensor<Q> values2coeffs(const keyT& key, const Tensor<Q>& values) const {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            double scale = pow(0.5,0.5*NDIM*key.level())*sqrt(FunctionDefaults<NDIM>::get_cell_volume());
            return transform(values,cdata.quad_phiw).scale(scale);
        }

        /// Compute the function values for multiplication

        /// Given coefficients from a parent cell, compute the value of
        /// the functions at the quadrature points of a child
        template <typename Q>
        Tensor<Q> fcube_for_mul(const keyT& child, const keyT& parent, const Tensor<Q>& coeff) const {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            if (child.level() == parent.level()) {
                return coeffs2values(parent, coeff);
            }
            else if (child.level() < parent.level()) {
                MADNESS_EXCEPTION("FunctionImpl: fcube_for_mul: child-parent relationship bad?",0);
            }
            else {
                Tensor<double> phi[NDIM];
                for (std::size_t d=0; d<NDIM; ++d) {
                    phi[d] = Tensor<double>(cdata.k,cdata.npt);
                    phi_for_mul(parent.level(),parent.translation()[d],
                                child.level(), child.translation()[d], phi[d]);
                }
                return general_transform(coeff,phi).scale(1.0/sqrt(FunctionDefaults<NDIM>::get_cell_volume()));;
            }
        }


        /// Invoked as a task by mul with the actual coefficients
        template <typename L, typename R>
        Void do_mul(const keyT& key, const Tensor<L>& left, const std::pair< keyT, Tensor<R> >& arg) {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            const keyT& rkey = arg.first;
            const Tensor<R>& rcoeff = arg.second;
            if(DEBUG && rkey == debugKey){
		madness::print("do_mul: r", rkey, rcoeff.normf());
		madness::print("do_mul: l", key, left.normf());
	    }
	
	    
            Tensor<R> rcube = fcube_for_mul(key, rkey, rcoeff);
            //madness::print("do_mul: l", key, left.size());
            Tensor<L> lcube = fcube_for_mul(key, key, left);
	    
	    
            Tensor<T> tcube(cdata.vk,false);
            TERNARY_OPTIMIZED_ITERATOR(T, tcube, L, lcube, R, rcube, *_p0 = *_p1 * *_p2;);
            double scale = pow(0.5,0.5*NDIM*key.level())*sqrt(FunctionDefaults<NDIM>::get_cell_volume());
            tcube = transform(tcube,cdata.quad_phiw).scale(scale);
	    if(DEBUG && key == debugKey){
		std::cout<<"Computed Coeff at "<<key<<" Norm is :"<<tcube.normf()<<std::endl;
	    }
	
            coeffs.replace(key, nodeT(coeffT(tcube),false));
            return None;
        }

        /// Invoked by result to perform result += alpha*left+beta*right in wavelet basis

        /// Does not assume that any of result, left, right have the same distribution.
        /// For most purposes result will start as an empty so actually are implementing
        /// out of place gaxpy.  If all functions have the same distribution there is
        /// no communication except for the optional fence.
        template <typename L, typename R>
        void gaxpy(T alpha, const FunctionImpl<L,NDIM>& left,
                   T beta, const FunctionImpl<R,NDIM>& right, bool fence) {
            // Loop over local nodes in both functions.  Add in left and subtract right.
            // Not that efficient in terms of memory bandwidth but ensures we do
            // not miss any nodes.
            typename FunctionImpl<L,NDIM>::dcT::const_iterator left_end = left.coeffs.end();
            for (typename FunctionImpl<L,NDIM>::dcT::const_iterator it=left.coeffs.begin();
                    it!=left_end;
                    ++it) {
                const keyT& key = it->first;
                const typename FunctionImpl<L,NDIM>::nodeT& other_node = it->second;
                coeffs.send(key, &nodeT:: template gaxpy_inplace<T,L>, 1.0, other_node, alpha);
            }
            typename FunctionImpl<R,NDIM>::dcT::const_iterator right_end = right.coeffs.end();
            for (typename FunctionImpl<R,NDIM>::dcT::const_iterator it=right.coeffs.begin();
                    it!=right_end;
                    ++it) {
                const keyT& key = it->first;
                const typename FunctionImpl<L,NDIM>::nodeT& other_node = it->second;
                coeffs.send(key, &nodeT:: template gaxpy_inplace<T,R>, 1.0, other_node, beta);
            }
            if (fence)
                world.gop.fence();
        }


        // Multiplication using recursive descent and assuming same distribution
        template <typename L, typename R>
        Void mulXXa(const keyT& key,
                    const FunctionImpl<L,NDIM>* left, const Tensor<L>& lcin,
                    const FunctionImpl<R,NDIM>* right,const Tensor<R>& rcin,
                    double tol) {
            typedef typename FunctionImpl<L,NDIM>::dcT::const_iterator literT;
            typedef typename FunctionImpl<R,NDIM>::dcT::const_iterator riterT;

            double lnorm=1e99, rnorm=1e99;

            Tensor<L> lc = lcin;
            if (lc.size() == 0) {
                literT it = left->coeffs.find(key).get();
                MADNESS_ASSERT(it != left->coeffs.end());
                lnorm = it->second.get_norm_tree();
                if (it->second.has_coeff())
                    lc = copy(it->second.coeff());
            }

            Tensor<R> rc = rcin;
            if (rc.size() == 0) {
                riterT it = right->coeffs.find(key).get();
                MADNESS_ASSERT(it != right->coeffs.end());
                rnorm = it->second.get_norm_tree();
                if (it->second.has_coeff())
                    rc = copy(it->second.coeff());
            }

            // both nodes are leaf nodes: multiply and return
            if (rc.size() && lc.size()) { // Yipee!
                do_mul<L,R>(key, lc, std::make_pair(key,rc));
                return None;
            }

            if (tol) {
                if (lc.size())
                    lnorm = lc.normf(); // Otherwise got from norm tree above
                if (rc.size())
                    rnorm = rc.normf();
                if (lnorm*rnorm < truncate_tol(tol, key)) {
                    coeffs.replace(key, nodeT(coeffT(cdata.vk),false)); // Zero leaf node
                    return None;
                }
            }

            // Recur down
            coeffs.replace(key, nodeT(coeffT(),true)); // Interior node

            Tensor<L> lss;
            if (lc.size()) {
                Tensor<L> ld(cdata.v2k);
                ld(cdata.s0) = lc(___);
                lss = left->unfilter(ld);
            }

            Tensor<R> rss;
            if (rc.size()) {
                Tensor<R> rd(cdata.v2k);
                rd(cdata.s0) = rc(___);
                rss = right->unfilter(rd);
            }

            for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                const keyT& child = kit.key();
                Tensor<L> ll;
                Tensor<R> rr;
                if (lc.size())
                    ll = copy(lss(child_patch(child)));
                if (rc.size())
                    rr = copy(rss(child_patch(child)));

                woT::task(coeffs.owner(child), &implT:: template mulXXa<L,R>, child, left, ll, right, rr, tol);
            }

            return None;
        }

        template <typename L, typename R>
        void mulXX(const FunctionImpl<L,NDIM>* left, const FunctionImpl<R,NDIM>* right, double tol, bool fence) {
            if (world.rank() == coeffs.owner(cdata.key0))
                mulXXa(cdata.key0, left, Tensor<L>(), right, Tensor<R>(), tol);
            if (fence)
                world.gop.fence();

            //verify_tree();
        }

        /// Verify tree is properly constructed ... global synchronization involved

        /// If an inconsistency is detected, prints a message describing the error and
        /// then throws a madness exception.
        ///
        /// This is a reasonably quick and scalable operation that is
        /// useful for debugging and paranoia.
        void verify_tree() const;

        const Tensor<T> parent_to_child(const coeffT& s, const keyT& parent, const keyT& child) const;
        T eval_cube(Level n, coordT& x, const tensorT& c) const;

        /// Evaluate the function at a point in \em simulation coordinates

        /// Only the invoking process will get the result via the
        /// remote reference to a future.  Active messages may be sent
        /// to other nodes.
        Void eval(const Vector<double,NDIM>& xin,
                  const keyT& keyin,
                  const typename Future<T>::remote_refT& ref);


        /// Computes norm of low/high-order polyn. coeffs for autorefinement test

        /// t is a k^d tensor.  In order to screen the autorefinement
        /// during multiplication compute the norms of
        /// ... lo ... the block of t for all polynomials of order < k/2
        /// ... hi ... the block of t for all polynomials of order >= k/2
        ///
        /// k=5   0,1,2,3,4     --> 0,1,2 ... 3,4
        /// k=6   0,1,2,3,4,5   --> 0,1,2 ... 3,4,5
        ///
        /// k=number of wavelets, so k=5 means max order is 4, so max exactly
        /// representable squarable polynomial is of order 2.
        void tnorm(const tensorT& t, double* lo, double* hi) const;


        /// Returns true if this block of coeffs needs autorefining
        bool autorefine_square_test(const keyT& key, const nodeT& t) const {
            double lo, hi;
            tnorm(t.coeff(), &lo, &hi);
            double test = 2*lo*hi + hi*hi;
            //print("autoreftest",key,thresh,truncate_tol(thresh, key),lo,hi,test);
            return test> truncate_tol(thresh, key);
        }

        /// Transform sum coefficients at level n to sums+differences at level n-1

        /// Given scaling function coefficients s[n][l][i] and s[n][l+1][i]
        /// return the scaling function and wavelet coefficients at the
        /// coarser level.  I.e., decompose Vn using Vn = Vn-1 + Wn-1.
        /// \code
        /// s_i = sum(j) h0_ij*s0_j + h1_ij*s1_j
        /// d_i = sum(j) g0_ij*s0_j + g1_ij*s1_j
        //  \endcode
        /// Returns a new tensor and has no side effects.  Works for any
        /// number of dimensions.
        ///
        /// No communication involved.
        tensorT filter(const tensorT& s) const {
            tensorT r(cdata.v2k,false);
            tensorT w(cdata.v2k,false);
            return fast_transform(s,cdata.hgT,r,w);
            //return transform(s,cdata.hgT);
        }

        ///  Transform sums+differences at level n to sum coefficients at level n+1

        ///  Given scaling function and wavelet coefficients (s and d)
        ///  returns the scaling function coefficients at the next finer
        ///  level.  I.e., reconstruct Vn using Vn = Vn-1 + Wn-1.
        ///  \code
        ///  s0 = sum(j) h0_ji*s_j + g0_ji*d_j
        ///  s1 = sum(j) h1_ji*s_j + g1_ji*d_j
        ///  \endcode
        ///  Returns a new tensor and has no side effects
        ///
        ///  If (sonly) ... then ss is only the scaling function coeff (and
        ///  assume the d are zero).  Works for any number of dimensions.
        ///
        /// No communication involved.
        tensorT unfilter(const tensorT& s) const {
            tensorT r(cdata.v2k,false);
            tensorT w(cdata.v2k,false);
            return fast_transform(s,cdata.hg,r,w);
            //return transform(s, cdata.hg);
        }

        void reconstruct(bool fence) {
            // Must set true here so that successive calls without fence do the right thing
            MADNESS_ASSERT(not is_redundant());
            nonstandard = compressed = redundant = false;
            if (world.rank() == coeffs.owner(cdata.key0))
                woT::task(world.rank(), &implT::reconstruct_op, cdata.key0,coeffT());
            if (fence)
                world.gop.fence();
        }

        Void reconstruct_op(const keyT& key, const coeffT& s);

        /// compress the wave function

        /// after application there will be sum coefficients at the root level,
        /// and difference coefficients at all other levels; furthermore:
        /// @param[in] nonstandard	keep sum coeffs at all other levels, except leaves
        /// @param[in] keepleaves	keep sum coeffs (but no diff coeffs) at leaves
        /// @param[in] redundant    keep only sum coeffs at all levels, discard difference coeffs
        void compress(bool nonstandard, bool keepleaves, bool redundant, bool fence) {
            MADNESS_ASSERT(not is_redundant());
            // Must set true here so that successive calls without fence do the right thing
            this->compressed = true;
            this->nonstandard = nonstandard;
            this->redundant = redundant;

            // these two are exclusive
            MADNESS_ASSERT(not (redundant and nonstandard));
            // otherwise we loose information
            if (redundant) {MADNESS_ASSERT(keepleaves);}

//            this->print_tree();
            if (world.rank() == coeffs.owner(cdata.key0)) {

           		compress_spawn(cdata.key0, nonstandard, keepleaves, redundant);
            }
            if (fence)
                world.gop.fence();
        }

        // Invoked on node where key is local
        Future<coeffT > compress_spawn(const keyT& key, bool nonstandard, bool keepleaves, bool redundant);


        /// calculate the wavelet coefficients using the sum coefficients of all child nodes

        /// @param[in] key 	this's key
        /// @param[in] v 	sum coefficients of the child nodes
        /// @param[in] nonstandard  keep the sum coefficients with the wavelet coefficients
        /// @param[in] redundant    keep only the sum coefficients, discard the wavelet coefficients
        /// @return 		the sum coefficients
        coeffT compress_op(const keyT& key, const std::vector< Future<coeffT > >& v, bool nonstandard, bool redundant) {
            PROFILE_MEMBER_FUNC(FunctionImpl);

            MADNESS_ASSERT(not redundant);
            // Copy child scaling coeffs into contiguous block
            tensorT d(cdata.v2k);
//            coeffT d(cdata.v2k,targs);
            int i=0;
            for (KeyChildIterator<NDIM> kit(key); kit; ++kit,++i) {
//                d(child_patch(kit.key())) += v[i].get();
                d(child_patch(kit.key())) += copy(v[i].get());
            }

            d = filter(d);

            typename dcT::accessor acc;
            MADNESS_ASSERT(coeffs.find(acc, key));

            if (acc->second.has_coeff()) {
            	print(" stuff in compress_op");
//                const coeffT& c = acc->second.coeff();
                const tensorT c = copy(acc->second.coeff());
                if (c.dim(0) == k) {
                    d(cdata.s0) += c;
                }
                else {
                    d += c;
                }
            }

            // need the deep copy for contiguity
            coeffT ss=copy(d(cdata.s0));

            if (key.level()> 0 && !nonstandard)
                d(cdata.s0) = 0.0;

            // insert either sum or difference coefficients
            if (redundant) {
                acc->second.set_coeff(ss);
            } else {
                coeffT dd=coeffT(d);
                acc->second.set_coeff(dd);
            }

            // return sum coefficients
            return ss;
        }

        struct do_norm2sq_local {
            double operator()(typename dcT::const_iterator& it) const {
                const nodeT& node = it->second;
                if (node.has_coeff()) {
                    double norm = node.coeff().normf();
                    return norm*norm;
                }
                else {
                    return 0.0;
                }
            }

            double operator()(double a, double b) const {
                return (a+b);
            }

            template <typename Archive> void serialize(const Archive& ar) {
                throw "NOT IMPLEMENTED";
            }
        };


        /// Returns the square of the local norm ... no comms
        double norm2sq_local() const {
            PROFILE_MEMBER_FUNC(FunctionImpl);
            typedef Range<typename dcT::const_iterator> rangeT;
            return world.taskq.reduce<double,rangeT,do_norm2sq_local>(rangeT(coeffs.begin(),coeffs.end()),
                    do_norm2sq_local());
        }

        /// Returns the number of coefficients in the function ... collective global sum
        std::size_t size() const {
            std::size_t sum = 0;
            typename dcT::const_iterator end = coeffs.end();
            for (typename dcT::const_iterator it=coeffs.begin(); it!=end; ++it) {
                const nodeT& node = it->second;
                if (node.has_coeff())
                    sum+=node.size();
            }
            world.gop.sum(sum);

            return sum;
        }

    };

    template<typename T, std::size_t NDIM> Key<NDIM> FunctionImpl<T,NDIM>::debugKey=Key<NDIM>(0);


    namespace archive {
        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveLoadImpl<Archive,const FunctionImpl<T,NDIM>*> {
            static void load(const Archive& ar, const FunctionImpl<T,NDIM>*& ptr) {
                bool exists=false;
                ar & exists;
                if (exists) {
                    uniqueidT id;
                    ar & id;
                    World* world = World::world_from_id(id.get_world_id());
                    MADNESS_ASSERT(world);
                    ptr = static_cast< const FunctionImpl<T,NDIM>*>(world->ptr_from_id< WorldObject< FunctionImpl<T,NDIM> > >(id));
                    if (!ptr)
                        MADNESS_EXCEPTION("FunctionImpl: remote operation attempting to use a locally uninitialized object",0);
                } else {
                    ptr=NULL;
                }
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveStoreImpl<Archive,const FunctionImpl<T,NDIM>*> {
            static void store(const Archive& ar, const FunctionImpl<T,NDIM>*const& ptr) {
                bool exists=(ptr) ? true : false;
                ar & exists;
                if (exists) ar & ptr->id();
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveLoadImpl<Archive, FunctionImpl<T,NDIM>*> {
            static void load(const Archive& ar, FunctionImpl<T,NDIM>*& ptr) {
                bool exists=false;
                ar & exists;
                if (exists) {
                    uniqueidT id;
                    ar & id;
                    World* world = World::world_from_id(id.get_world_id());
                    MADNESS_ASSERT(world);
                    ptr = static_cast< FunctionImpl<T,NDIM>*>(world->ptr_from_id< WorldObject< FunctionImpl<T,NDIM> > >(id));
                    if (!ptr)
                        MADNESS_EXCEPTION("FunctionImpl: remote operation attempting to use a locally uninitialized object",0);
                } else {
                    ptr=NULL;
                }
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveStoreImpl<Archive, FunctionImpl<T,NDIM>*> {
            static void store(const Archive& ar, FunctionImpl<T,NDIM>*const& ptr) {
                bool exists=(ptr) ? true : false;
                ar & exists;
                if (exists) ar & ptr->id();
//                ar & ptr->id();
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveLoadImpl<Archive, std::shared_ptr<const FunctionImpl<T,NDIM> > > {
            static void load(const Archive& ar, std::shared_ptr<const FunctionImpl<T,NDIM> >& ptr) {
                const FunctionImpl<T,NDIM>* f = NULL;
                ArchiveLoadImpl<Archive, const FunctionImpl<T,NDIM>*>::load(ar, f);
                ptr.reset(f, & madness::detail::no_delete<const FunctionImpl<T,NDIM> >);
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveStoreImpl<Archive, std::shared_ptr<const FunctionImpl<T,NDIM> > > {
            static void store(const Archive& ar, const std::shared_ptr<const FunctionImpl<T,NDIM> >& ptr) {
                ArchiveStoreImpl<Archive, const FunctionImpl<T,NDIM>*>::store(ar, ptr.get());
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveLoadImpl<Archive, std::shared_ptr<FunctionImpl<T,NDIM> > > {
            static void load(const Archive& ar, std::shared_ptr<FunctionImpl<T,NDIM> >& ptr) {
                FunctionImpl<T,NDIM>* f = NULL;
                ArchiveLoadImpl<Archive, FunctionImpl<T,NDIM>*>::load(ar, f);
                ptr.reset(f, & madness::detail::no_delete<FunctionImpl<T,NDIM> >);
            }
        };

        template <class Archive, class T, std::size_t NDIM>
        struct ArchiveStoreImpl<Archive, std::shared_ptr<FunctionImpl<T,NDIM> > > {
            static void store(const Archive& ar, const std::shared_ptr<FunctionImpl<T,NDIM> >& ptr) {
                ArchiveStoreImpl<Archive, FunctionImpl<T,NDIM>*>::store(ar, ptr.get());
            }
        };
    }

}

#endif // MADNESS_MRA_FUNCIMPL_H__INCLUDED
