/*
  This file is part of MADNESS.
  
  Copyright (C) <2007> <Oak Ridge National Laboratory>
  
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

/// \file loadbal.h
/// \brief Declares and partially implements MyPmap, LoadBalImpl and associated load balancing classes.

  
#ifndef LOADBAL_H
#define LOADBAL_H

namespace madness {

    typedef int Cost;
    typedef double CompCost;

    /// Finds exponent k such that d^k <= me < d^{k+1}
    inline int nearest_power(int me, int d) {
        int k = 0;
        while (me != 0) {
            if (me%d == 0) {
                k++;
                me/=d;
            } else {
                break;
            }
        }
        return k;
    };

    template <typename Data, int D> class LBNode;
    template <int D> struct TreeCoords;
    template <int D> class MyPmap;
    template <int D> class LBTree;
    class NodeData;

    /// Convenient typedef shortcuts

    /// Makes it easier to handle these unwieldy templated types
    template <int D>
    struct DClass {
        typedef Key<D> KeyD;
        typedef const Key<D> KeyDConst;
        typedef TreeCoords<D> TreeCoords;
        typedef LBNode<NodeData,D> NodeD;
        typedef const LBNode<NodeData,D> NodeDConst;
        typedef MyPmap<D> MyPmap;
        typedef LBTree<D> treeT;
    };


    /// The node that is used in the fascimile copy of the tree to be load balanced

    /// The node used in the tree that is operated upon and load balanced in LoadBalImpl.
    template <typename Data, int D>
    class LBNode {
    private:
        Data data;
        std::vector<bool> c; /// Existence of each child individually

        void all_children(bool status=false) {
            c.clear();
            c.assign(dim, status);
        };

    public:
        static int dim; /// Number of children in standard tree (e.g. 2^D)
	int nrecvd;

        LBNode() {
            data = Data();
            all_children();
	    nrecvd = 0;
        };

        LBNode(const Data& d, bool children=false, int n=0) : data(d), nrecvd(n) {
            all_children(children);
        };

	/// Determines whether node has any children at all
        bool has_children() const {
            for (int i = 0; i < dim; i++)
                if (c[i]) return true;
            return false;
        };

        bool has_child(unsigned int i) const {
            return c[i];
        };

        bool has_child(int i) const {
            return c[i];
        };

	int get_num_children() const {
	    int nkids = 0;
	    for (int i=0; i < dim; i++) {
		if (has_child(i)) nkids++;
	    }
	    return nkids;
	}

        void set_child(int i, bool setto = true) {
            c[i] = setto;
        };

        void set_data(const Data& d) {
            data = d;
        };

        Data get_data() const {
            return data;
        };

        vector<bool> get_c() const {
            return c;
        };

        template <typename Archive>
        void serialize(const Archive& ar) {
            ar & data & c;
        }
    };


    template <typename Data, int D>
    std::ostream& operator<<(std::ostream& s, const LBNode<Data, D>& node) {
        s << "data = " << node.get_data() << ", c = " << node.get_c();
        return s;
    };

    template <int D>
    std::ostream& operator<<(std::ostream& s, typename DClass<D>::NodeDConst& node) {
        s << "data = " << node.get_data() << ", c = " << node.get_c();
        return s;
    };


    template <typename Data, int D>
    int LBNode<Data,D>::dim = power<D>();


    /// Diagnostic data contained in fascimile tree
    /// Diagnostic data, including the cost of the node and the subtree headed by that node,
    /// along with a bool flag used during depth-first partitioning
    class NodeData {
        friend std::ostream& operator<<(std::ostream& s, const NodeData& nd);
    public:
        int cost;
        int subcost;
        bool is_taken;
        NodeData(int c = 1, int s = 1, bool i = false, int n = 0) : cost(c), subcost(s), is_taken(i) {};
        template <typename Archive>
        void serialize(const Archive& ar) {
            ar & cost & subcost & is_taken;
        }
        void print() {
            cout << "cost = " << cost << ", subcost = " << subcost << ", is_taken = " << is_taken << endl;
        }
    };


    inline std::ostream& operator<<(std::ostream& s, const NodeData& nd) {
        s << "cost " << nd.cost << ", subcost " << nd.subcost << ", is_taken " << nd.is_taken;
        return s;
    };



    /// Key + owner, struct used to determine mapping of tree nodes
    template <int D>
    struct TreeCoords {
        Key<D> key;
        ProcessID owner;

        TreeCoords(const Key<D>& k, ProcessID o) : key(Key<D>(k)), owner(o) {};
        TreeCoords(const TreeCoords& t) : key(Key<D>(t.key)), owner(t.owner) {};
        TreeCoords() : key(Key<D>()), owner(-1) {};
        void print() const {
            madness::print(key, "   owner =", owner);
        };

        bool operator< (const TreeCoords t) const {
            return (this->key < t.key);
        };
    };


    template<int D>
    class ProcMapImpl {
	public:
/*
#ifdef WORLDDC_USES_GNU_HASH_MAP
        template <typename T>
        struct PMLocalHash {
            std::size_t operator()(const T& t) const {
                return hash(t);
            };
        };
        typedef HASH_MAP_NAMESPACE::hash_map< typename DClass<D>::KeyDConst,ProcessID,PMLocalHash<typename DClass<D>::KeyDConst > > Mapinfo;
#else
*/
        typedef std::map<typename DClass<D>::KeyDConst,ProcessID> Mapinfo;
/*
#endif
*/
	typedef typename Mapinfo::iterator iterator;
	typedef const iterator iterator_const;
	typedef std::pair< typename DClass<D>::KeyDConst, ProcessID > pairT;

	ProcMapImpl() {};
	ProcMapImpl(std::vector< TreeCoords<D> > v) {
	    int vlen = v.size();
	    for (int i = 0; i < vlen; i++) {
	    	themap.insert(std::make_pair(v[i].key, v[i].owner));
	    }
	};

	ProcMapImpl(const TreeCoords<D>& t) {
	    themap.insert(std::make_pair(t.key, t.owner));
	};
	void insert(const TreeCoords<D>& t) {
	    themap.insert(std::make_pair(t.key, t.owner));
	};
	void erase(const TreeCoords<D>& t) {
	    themap.erase(t.key);
	};

        ProcessID find_owner(const Key<D>& key) const {
//	    iterator_const it = themap.find(key);
	    typename std::map<typename DClass<D>::KeyDConst,ProcessID>::const_iterator it = themap.find(key);
	    if (it != themap.end()) {
		return it->second;
	    } else if (key.level() == 0) {
		madness::print("find_owner: owner of ", key, "not found but returning 0");
		return 0;
	    } else {
		return this->find_owner(key.parent());
	    }
	};

	void print() {
	    for (iterator it = themap.begin(); it != themap.end(); ++it) {
		madness::print(it->first, "   ", it->second);
	    }
	}

	private:
	Mapinfo themap;
    };


    /// Procmap implemented using Tree of TreeCoords

    template <int D>
    class MyPmap : public WorldDCPmapInterface< Key<D> > {
    private:
        bool simplemap;
	const int nproc;
        const ProcessID me;
	const int n;
	SharedPtr< ProcMapImpl<D> > tree_map;
        typedef Key<D> KeyD;

	/// private method that builds the Tree underlying the procmap
	void build_tree_map(std::vector< TreeCoords<D> > v) {
	    tree_map = SharedPtr< ProcMapImpl<D> > (new ProcMapImpl<D>(v));
	};

	ProcessID simple_hash(const KeyD& key) const {
	    if (key.level() == 0) {
		return 0;
	    } else if (key.level() <= n) {
//		return hash(key)%nproc;
		return (key.hash()%nproc);
	    } else {
//		return hash(key.parent(key.level()-n))%nproc;
		return ((key.parent(key.level()-n)).hash()%nproc);
	    }
	};

    public:
        MyPmap() : simplemap(false) {};
	/*
        MyPmap(World& world) : simplemap(false), nproc(world.nproc()), me(world.rank()), n(0) {
            int twotoD = power<D>();
            const int level = nearest_power(nproc, twotoD);
            int NPin = (int) pow((double)twotoD,level);
            vector<TreeCoords<D> > v;

            for (Translation i=0; i < (Translation)NPin; i++) {
                KeyD key(level,i);
                if ((i%twotoD) == 0) {
                    key = key.parent(nearest_power(NPin-i, twotoD));
                }
                v.push_back(TreeCoords<D>(key,i));
            }
            build_tree_map(v);
//            madness::print("MyPmap constructor");
//            tree_map->print();
        };
	*/
	MyPmap(World& world) : simplemap(true), nproc(world.nproc()), me(world.rank()), n(100) {};

        MyPmap(World& world, int n) : simplemap(true), nproc(world.nproc()), me(world.rank()), n(n) {};

        MyPmap(World& world, vector<TreeCoords<D> > v) : simplemap(false), nproc(world.nproc()), me(world.rank()), n(0) {
            build_tree_map(v);
//            madness::print("");
//            tree_map->print();
        };

        MyPmap(const MyPmap<D>& other) : simplemap(other.staticmap), nproc(other.nproc), me(other.me), n(other.n), tree_map(other.tree_map) {};

        MyPmap<D>& operator=(const MyPmap<D>& other) {
            if (this != &other) {
                simplemap = other.simplemap;
                nproc = other.nproc;
		me = other.me;
		n = other.n;
                tree_map = other.tree_map;
            }
            return *this;
        };

        void print() const {
	    if (!simplemap) {
		tree_map->print();
	    } else {
		madness::print("MyPmap: simple map with n =", n);
	    }
        };

	/// Find the owner of a given key
        ProcessID owner(const KeyD& key) const {
            if (simplemap)
                return simple_hash(key);
            else {
		return tree_map->find_owner(key);
            }
        };
    };

    /// The container in which the fascimile tree with its keys mapping to LBNodes is stored
    template <int D>
    class LBTree : public WorldObject< LBTree<D> > {
    public:
	typedef WorldObject<LBTree<D> > woT;
        typedef WorldContainer<typename DClass<D>::KeyD,typename DClass<D>::NodeD> dcT;
	typedef typename dcT::iterator iterator;
	World& world;

    private:
        dcT impl;

    public:
        //LBTree() {};
        LBTree(World& world, const SharedPtr< WorldDCPmapInterface<typename DClass<D>::KeyD> >& pmap) : woT(world)
	    , world(world), impl(world,pmap) {
        };
	/// Initialize the LBTree by converting a FunctionImpl to a LBTree
        template <typename T>
        inline void init_tree(const SharedPtr< FunctionImpl<T,D> >& f) {
            for (typename FunctionImpl<T,D>::dcT::iterator it = f->coeffs.begin(); it != f->coeffs.end(); ++it) {
            	// convert Node to LBNode
            	NodeData nd;
		typename DClass<D>::KeyD key = it->first;
            	if (!(it->second.has_children())) {
                	typename DClass<D>::NodeD lbnode(nd,false);
                	// insert into impl
                	impl.insert(key, lbnode);
            	} else {
                	typename DClass<D>::NodeD lbnode(nd,true);
                	// insert into impl
                	impl.insert(key, lbnode);
                }
            }
        };

        // Methods:
        void print(typename DClass<D>::KeyDConst& key) {
            typename DClass<D>::treeT::iterator it = impl.find(key);
            if (it == impl.end()) return;
            for (Level i = 0; i < key.level(); i++) cout << "  ";
            madness::print(key, it->second);
            for (KeyChildIterator<D> kit(key); kit; ++kit) {
                print(kit.key());
            }
        };

        std::vector< std::vector<typename DClass<D>::TreeCoords> > find_partitions(SharedPtr<std::vector<Cost> > costlist);

        void fix_cost();

	void init_fix_cost();
	void fix_cost_spawn();
	Void fix_cost_sum(typename DClass<D>::KeyDConst& key, Cost c);
/*
	Future<Cost> fix_cost_spawn(typename DClass<D>::KeyDConst& key);
	Cost fix_cost_add_op(typename DClass<D>::KeyDConst& key, const std::vector< Future<Cost> >& vcost);
*/
        Cost depth_first_partition(typename DClass<D>::KeyDConst& key,
                                 vector<typename DClass<D>::TreeCoords>* klist, unsigned int npieces,
                                 Cost totalcost = 0, Cost *maxcost = 0);

        void rollup();

	void reset(bool taken);

        void meld(typename DClass<D>::treeT::iterator it);

        Cost make_partition(typename DClass<D>::KeyDConst& key,
                           std::vector<typename DClass<D>::KeyD>* klist, Cost partition_size,
                           bool last_partition, Cost used_up, bool *atleaf);

        void remove_cost(typename DClass<D>::KeyDConst& key, Cost c);

        Cost compute_cost(typename DClass<D>::KeyDConst& key);


        MyPmap<D>& get_mypmap() {
            return *static_cast< MyPmap<D>* >(impl.get_pmap().get());
        };

    };

    /// Implementation of load balancing

    /// Implements the load balancing algorithm upon the tree underlying a function.
    template <typename T, int D>
    class LoadBalImpl {
    private:
	typedef MyPmap<D> Pmap;
        Function<T,D> f;
	double comm_bandw;
	double comm_latency;
	double flop_time;
        SharedPtr<typename DClass<D>::treeT> skeltree;

        void construct_skel(SharedPtr<FunctionImpl<T,D> > f) {
            skeltree = SharedPtr<typename DClass<D>::treeT>(new typename DClass<D>::treeT(f->world,
                       f->coeffs.get_pmap()));
	    //            typename DClass<D>::KeyD root(0);
//            madness::print("about to initialize tree");
	    skeltree->template init_tree<T>(f);
//            madness::print("just initialized tree");
        };

    public:
        //Constructors
        LoadBalImpl() {};

        LoadBalImpl(Function<T,D> f, double a=1e-8, double b=1e-5, double c=5e-10) : f(f), comm_bandw(a), comm_latency(b), flop_time(c) {
            construct_skel(f.get_impl());
        };

        ~LoadBalImpl() {};

        //Methods

	/// Returns a shared pointer to a new process map, which can then be used to redistribute the function
        SharedPtr< WorldDCPmapInterface< Key<D> > > load_balance() {
            return SharedPtr< WorldDCPmapInterface< Key<D> > >(new MyPmap<D>(f.get_impl()->world, find_best_partition()));
        };

        vector<typename DClass<D>::TreeCoords> find_best_partition();

	CompCost compute_comp_cost(Cost c, int n);
    };


    Cost compute_partition_size(Cost cost, unsigned int parts);

}

#endif
