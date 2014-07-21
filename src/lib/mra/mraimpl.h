
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

#ifndef MADNESS_MRA_MRAIMPL_H__INCLUDED
#define MADNESS_MRA_MRAIMPL_H__INCLUDED

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <math.h>
#include <world/worldhashmap.h>
#include <mra/function_common_data.h>

#include <mra/funcimpl.h>
#include <mra/displacements.h>

/// \file mra/mraimpl.h
/// \brief Declaration and initialization of static data, some implementation, some instantiation

namespace madness {

    // Definition and initialization of FunctionDefaults static members
    // It cannot be an instance of FunctionFactory since we want to
    // set the defaults independent of the data type.

    template <typename T, std::size_t NDIM>
    void FunctionCommonData<T,NDIM>::_init_twoscale() {
        if (! two_scale_hg(k, &hg)) throw "failed to get twoscale coefficients";
        hgT = copy(transpose(hg));

        Slice sk(0,k-1), sk2(k,-1);
        hgsonly = copy(hg(Slice(0,k-1),_));

        h0 = copy(hg(sk,sk));
        h1 = copy(hg(sk,sk2));
        g0 = copy(hg(sk2,sk));
        g1 = copy(hg(sk2,sk2));

        h0T = copy(transpose(hg(sk,sk)));
        h1T = copy(transpose(hg(sk,sk2)));
        g0T = copy(transpose(hg(sk2,sk)));
        g1T = copy(transpose(hg(sk2,sk2)));

    }

    template <typename T, std::size_t NDIM>
    void FunctionCommonData<T,NDIM>::_init_quadrature
    (int k, int npt, Tensor<double>& quad_x, Tensor<double>& quad_w,
     Tensor<double>& quad_phi, Tensor<double>& quad_phiw, Tensor<double>& quad_phit) {
        quad_x = Tensor<double>(npt);
        quad_w = Tensor<double>(npt);
        quad_phi = Tensor<double>(npt,k);
        quad_phiw = Tensor<double>(npt,k);

        gauss_legendre(npt,0.0,1.0,quad_x.ptr(),quad_w.ptr());
        for (int mu=0; mu<npt; ++mu) {
            double phi[200];
            legendre_scaling_functions(quad_x(mu),k,phi);
            for (int j=0; j<k; ++j) {
                quad_phi(mu,j) = phi[j];
                quad_phiw(mu,j) = quad_w(mu)*phi[j];
            }
        }
        quad_phit = transpose(quad_phi);
    }


    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::verify_tree() const {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        world.gop.fence();  // Make sure nothing is going on

        // Verify consistency of compression status, existence and size of coefficients,
        // and has_children() flag.
        for (typename dcT::const_iterator it=coeffs.begin(); it!=coeffs.end(); ++it) {
            const keyT& key = it->first;
            const nodeT& node = it->second;
            bool bad;

            if (is_compressed()) {
                if (node.has_children()) {
                    bad = (node.coeff().has_data()) and (node.coeff().dim(0) != 2*cdata.k);
                }
                else {
//                    bad = node.coeff().size() != 0;
                    bad = node.coeff().has_data();
                }
            }
            else {
                if (node.has_children()) {
//                    bad = node.coeff().size() != 0;
                    bad = node.coeff().has_data();
                }
                else {
                    bad = (node.coeff().has_data()) and ( node.coeff().dim(0) != cdata.k);
                }
            }

            if (bad) {
                print(world.rank(), "FunctionImpl: verify: INCONSISTENT TREE NODE, key =", key, ", node =", node,
                      ", dim[0] =",node.coeff().dim(0),", compressed =",is_compressed());
                std::cout.flush();
                MADNESS_EXCEPTION("FunctionImpl: verify: INCONSISTENT TREE NODE", 0);
            }
        }

        // Ensure that parents and children exist appropriately
        for (typename dcT::const_iterator it=coeffs.begin(); it!=coeffs.end(); ++it) {
            const keyT& key = it->first;
            const nodeT& node = it->second;

            if (key.level() > 0) {
                const keyT parent = key.parent();
                typename dcT::const_iterator pit = coeffs.find(parent).get();
                if (pit == coeffs.end()) {
                    print(world.rank(), "FunctionImpl: verify: MISSING PARENT",key,parent);
                    std::cout.flush();
                    MADNESS_EXCEPTION("FunctionImpl: verify: MISSING PARENT", 0);
                }
                const nodeT& pnode = pit->second;
                if (!pnode.has_children()) {
                    print(world.rank(), "FunctionImpl: verify: PARENT THINKS IT HAS NO CHILDREN",key,parent);
                    std::cout.flush();
                    MADNESS_EXCEPTION("FunctionImpl: verify: PARENT THINKS IT HAS NO CHILDREN", 0);
                }
            }

            for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                typename dcT::const_iterator cit = coeffs.find(kit.key()).get();
                if (cit == coeffs.end()) {
                    if (node.has_children()) {
                        print(world.rank(), "FunctionImpl: verify: MISSING CHILD",key,kit.key());
                        std::cout.flush();
                        MADNESS_EXCEPTION("FunctionImpl: verify: MISSING CHILD", 0);
                    }
                }
                else {
                    if (! node.has_children()) {
                        print(world.rank(), "FunctionImpl: verify: UNEXPECTED CHILD",key,kit.key());
                        std::cout.flush();
                        MADNESS_EXCEPTION("FunctionImpl: verify: UNEXPECTED CHILD", 0);
                    }
                }
            }
        }

        world.gop.fence();
    }

    template <typename T, std::size_t NDIM>
    T FunctionImpl<T,NDIM>::eval_cube(Level n, coordT& x, const tensorT& c) const {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        const int k = cdata.k;
        double px[NDIM][k];
        T sum = T(0.0);

        for (std::size_t i=0; i<NDIM; ++i) legendre_scaling_functions(x[i],k,px[i]);

        if (NDIM == 1) {
            for (int p=0; p<k; ++p)
                sum += c(p)*px[0][p];
        }
        else if (NDIM == 2) {
            for (int p=0; p<k; ++p)
                for (int q=0; q<k; ++q)
                    sum += c(p,q)*px[0][p]*px[1][q];
        }
        else if (NDIM == 3) {
            for (int p=0; p<k; ++p)
                for (int q=0; q<k; ++q)
                    for (int r=0; r<k; ++r)
                        sum += c(p,q,r)*px[0][p]*px[1][q]*px[2][r];
        }
        else if (NDIM == 4) {
            for (int p=0; p<k; ++p)
                for (int q=0; q<k; ++q)
                    for (int r=0; r<k; ++r)
                        for (int s=0; s<k; ++s)
                            sum += c(p,q,r,s)*px[0][p]*px[1][q]*px[2][r]*px[3][s];
        }
        else if (NDIM == 5) {
            for (int p=0; p<k; ++p)
                for (int q=0; q<k; ++q)
                    for (int r=0; r<k; ++r)
                        for (int s=0; s<k; ++s)
                            for (int t=0; t<k; ++t)
                                sum += c(p,q,r,s,t)*px[0][p]*px[1][q]*px[2][r]*px[3][s]*px[4][t];
        }
        else if (NDIM == 6) {
            for (int p=0; p<k; ++p)
                for (int q=0; q<k; ++q)
                    for (int r=0; r<k; ++r)
                        for (int s=0; s<k; ++s)
                            for (int t=0; t<k; ++t)
                                for (int u=0; u<k; ++u)
                                    sum += c(p,q,r,s,t,u)*px[0][p]*px[1][q]*px[2][r]*px[3][s]*px[4][t]*px[5][u];
        }
        else {
            MADNESS_EXCEPTION("FunctionImpl:eval_cube:NDIM?",NDIM);
        }
        return sum*pow(2.0,0.5*NDIM*n)/sqrt(FunctionDefaults<NDIM>::get_cell_volume());
    }

    template <typename T, std::size_t NDIM>
    Void FunctionImpl<T,NDIM>::reconstruct_op(const keyT& key, const coeffT& s) {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        // Note that after application of an integral operator not all
        // siblings may be present so it is necessary to check existence
        // and if absent insert an empty leaf node.
        //
        // If summing the result of an integral operator (i.e., from
        // non-standard form) there will be significant scaling function
        // coefficients at all levels and possibly difference coefficients
        // in leaves, hence the tree may refine as a result.
        typename dcT::iterator it = coeffs.find(key).get();
        if (it == coeffs.end()) {
            coeffs.replace(key,nodeT(coeffT(),false));
            it = coeffs.find(key).get();
        }
        nodeT& node = it->second;

        // The integral operator will correctly connect interior nodes
        // to children but may leave interior nodes without coefficients
        // ... but they still need to sum down so just give them zeros
        if (node.has_children() && !node.has_coeff()) {
            node.set_coeff(coeffT(cdata.v2k));
        }

        if (node.has_children() || node.has_coeff()) { // Must allow for inconsistent state from transform, etc.
            coeffT d = node.coeff();
            if (!d.has_data()) d = coeffT(cdata.v2k);
            if (key.level() > 0) d(cdata.s0) += s; // -- note accumulate for NS summation
            if (d.dim(0)==2*get_k()) {              // d might be pre-truncated if it's a leaf
                d = unfilter(d);
                node.clear_coeff();
                node.set_has_children(true);
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    const keyT& child = kit.key();
                    coeffT ss = copy(d(child_patch(child)));
                    PROFILE_BLOCK(recon_send);
                    woT::task(coeffs.owner(child), &implT::reconstruct_op, child, ss);
                }
            } else {
                MADNESS_ASSERT(node.is_leaf());
                node.coeff()+=s;
                //node.coeff().reduce_rank(targs.thresh);
            }
        }
        else {
        	coeffT ss=s;
        	//if (s.has_no_data()) ss=coeffT(cdata.vk); ??????????????????????????
            if (key.level()) node.set_coeff(copy(ss));
            else node.set_coeff(ss);
        }
        return None;
    }

    template <typename T, std::size_t NDIM>
    Tensor<T> fcube(const Key<NDIM>& key, T (*f)(const Vector<double,NDIM>&), const Tensor<double>& qx) {
//      fcube(key,typename FunctionFactory<T,NDIM>::FunctorInterfaceWrapper(f) , qx, fval);
        std::vector<long> npt(NDIM,qx.dim(0));
        Tensor<T> fval(npt);
        fcube(key,ElementaryInterface<T,NDIM>(f) , qx, fval);
        return fval;
    }


    /// return the values of a Function on a grid

    /// @param[in]  key the key indicating where the quadrature points are located
    /// @param[in]  f   the interface to the elementary function
    /// @param[in]  qx  quadrature points on a level=0 box
    /// @param[out] fval    values
    template <typename T, std::size_t NDIM>
//    void FunctionImpl<T,NDIM>::fcube(const keyT& key, const FunctionFunctorInterface<T,NDIM>& f, const Tensor<double>& qx, tensorT& fval) const {
    void fcube(const Key<NDIM>& key, const FunctionFunctorInterface<T,NDIM>& f, const Tensor<double>& qx, Tensor<T>& fval) {
    //~ template <typename T, std::size_t NDIM> template< typename FF>
    //~ void FunctionImpl<T,NDIM>::fcube(const keyT& key, const FF& f, const Tensor<double>& qx, tensorT& fval) const {
        typedef Vector<double,NDIM> coordT;
        PROFILE_MEMBER_FUNC(FunctionImpl);
        const Vector<Translation,NDIM>& l = key.translation();
        const Level n = key.level();
        const double h = std::pow(0.5,double(n));
        coordT c; // will hold the point in user coordinates
        const int npt = qx.dim(0);

        const Tensor<double>& cell_width = FunctionDefaults<NDIM>::get_cell_width();
        const Tensor<double>& cell = FunctionDefaults<NDIM>::get_cell();

        if (NDIM == 1) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                fval(i) = f(c);
            }
        }
        else if (NDIM == 2) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                for (int j=0; j<npt; ++j) {
                    c[1] = cell(1,0) + h*cell_width[1]*(l[1] + qx(j)); // y
                    fval(i,j) = f(c);
                }
            }
        }
        else if (NDIM == 3) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                for (int j=0; j<npt; ++j) {
                    c[1] = cell(1,0) + h*cell_width[1]*(l[1] + qx(j)); // y
                    for (int k=0; k<npt; ++k) {
                        c[2] = cell(2,0) + h*cell_width[2]*(l[2] + qx(k)); // z
                        fval(i,j,k) = f(c);
                    }
                }
            }
        }
        else if (NDIM == 4) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                for (int j=0; j<npt; ++j) {
                    c[1] = cell(1,0) + h*cell_width[1]*(l[1] + qx(j)); // y
                    for (int k=0; k<npt; ++k) {
                        c[2] = cell(2,0) + h*cell_width[2]*(l[2] + qx(k)); // z
                        for (int m=0; m<npt; ++m) {
                            c[3] = cell(3,0) + h*cell_width[3]*(l[3] + qx(m)); // xx
                            fval(i,j,k,m) = f(c);
                        }
                    }
                }
            }
        }
        else if (NDIM == 5) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                for (int j=0; j<npt; ++j) {
                    c[1] = cell(1,0) + h*cell_width[1]*(l[1] + qx(j)); // y
                    for (int k=0; k<npt; ++k) {
                        c[2] = cell(2,0) + h*cell_width[2]*(l[2] + qx(k)); // z
                        for (int m=0; m<npt; ++m) {
                            c[3] = cell(3,0) + h*cell_width[3]*(l[3] + qx(m)); // xx
                            for (int n=0; n<npt; ++n) {
                                c[4] = cell(4,0) + h*cell_width[4]*(l[4] + qx(n)); // yy
                                fval(i,j,k,m,n) = f(c);
                            }
                        }
                    }
                }
            }
        }
        else if (NDIM == 6) {
            for (int i=0; i<npt; ++i) {
                c[0] = cell(0,0) + h*cell_width[0]*(l[0] + qx(i)); // x
                for (int j=0; j<npt; ++j) {
                    c[1] = cell(1,0) + h*cell_width[1]*(l[1] + qx(j)); // y
                    for (int k=0; k<npt; ++k) {
                        c[2] = cell(2,0) + h*cell_width[2]*(l[2] + qx(k)); // z
                        for (int m=0; m<npt; ++m) {
                            c[3] = cell(3,0) + h*cell_width[3]*(l[3] + qx(m)); // xx
                            for (int n=0; n<npt; ++n) {
                                c[4] = cell(4,0) + h*cell_width[4]*(l[4] + qx(n)); // yy
                                for (int p=0; p<npt; ++p) {
                                    c[5] = cell(5,0) + h*cell_width[5]*(l[5] + qx(p)); // zz
                                    fval(i,j,k,m,n,p) = f(c);
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            MADNESS_EXCEPTION("FunctionImpl: fcube: confused about NDIM?",NDIM);
        }
    }

    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::fcube(const keyT& key, T (*f)(const coordT&), const Tensor<double>& qx, tensorT& fval) const {
//      fcube(key,typename FunctionFactory<T,NDIM>::FunctorInterfaceWrapper(f) , qx, fval);
        madness::fcube(key,ElementaryInterface<T,NDIM>(f) , qx, fval);
    }

    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::fcube(const keyT& key, const FunctionFunctorInterface<T,NDIM>& f, const Tensor<double>& qx, tensorT& fval) const {
        madness::fcube(key,f,qx,fval);
    }


    /// project the functor into this functionimpl, and "return" a tree in reconstructed,
    /// rank-reduced form.

    /// @param[in]  key current FunctionNode
    /// @param[in]  do_refine
    /// @param[in]  specialpts  in case these are very spiky functions -- don't undersample
    template <typename T, std::size_t NDIM>
    Void FunctionImpl<T,NDIM>::project_refine_op(const keyT& key,
                                                 bool do_refine,
                                                 const std::vector<Vector<double,NDIM> >& specialpts) {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        if (do_refine && key.level() < max_refine_level) {

            // Restrict special points to this box
            std::vector<Vector<double,NDIM> > newspecialpts;
            if (key.level() < functor->special_level() && specialpts.size() > 0) {
                BoundaryConditions<NDIM> bc = FunctionDefaults<NDIM>::get_bc();
                std::vector<bool> bperiodic = bc.is_periodic();
                for (unsigned int i = 0; i < specialpts.size(); ++i) {
                    coordT simpt;
                    user_to_sim(specialpts[i], simpt);
                    Key<NDIM> specialkey = simpt2key(simpt, key.level());
                    if (specialkey.is_neighbor_of(key,bperiodic)) {
                        newspecialpts.push_back(specialpts[i]);
                    }
                }
            }

            // If refining compute scaling function coefficients and
            // norm of difference coefficients
            tensorT r, s0;
            double dnorm = 0.0;
            //////////////////////////if (newspecialpts.size() == 0)
            {
                // Make in r child scaling function coeffs at level n+1
                r = tensorT(cdata.v2k);
                for (KeyChildIterator<NDIM> it(key); it; ++it) {
                    const keyT& child = it.key();
                    r(child_patch(child)) = project(child);
                }
                // Filter then test difference coeffs at level n
                tensorT d = filter(r);
                if (truncate_on_project) s0 = copy(d(cdata.s0));
                d(cdata.s0) = T(0);
                dnorm = d.normf();
            }

            // If have special points always refine.  If don't have special points
            // refine if difference norm is big
            if (newspecialpts.size() > 0 || dnorm >=truncate_tol(thresh,key.level())) {
                coeffs.replace(key,nodeT(coeffT(),true)); // Insert empty node for parent
                for (KeyChildIterator<NDIM> it(key); it; ++it) {
                    const keyT& child = it.key();
                    ProcessID p;
                    if (FunctionDefaults<NDIM>::get_project_randomize()) {
                        p = world.random_proc();
                    }
                    else {
                        p = coeffs.owner(child);
                    }
                    PROFILE_BLOCK(proj_refine_send);
                    woT::task(p, &implT::project_refine_op, child, do_refine, newspecialpts);
                }
            }
            else {
                if (truncate_on_project) {
                	coeffT s(s0);
                    coeffs.replace(key,nodeT(s,false));
                }
                else {
                    coeffs.replace(key,nodeT(coeffT(),true)); // Insert empty node for parent
                    for (KeyChildIterator<NDIM> it(key); it; ++it) {
                        const keyT& child = it.key();
                        coeffT s(r(child_patch(child)));
                        coeffs.replace(child,nodeT(s,false));
                    }
                }
            }
        }
        else {
            coeffs.replace(key,nodeT(coeffT(project(key)),false));
        }
        return None;
    }


    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::insert_zero_down_to_initial_level(const keyT& key) {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        if (compressed) initial_level = std::max(initial_level,1); // Otherwise zero function is confused
        if (coeffs.is_local(key)) {
            if (compressed) {
                if (key.level() == initial_level) {
                    coeffs.replace(key, nodeT(coeffT(), false));
                }
                else {
                    coeffs.replace(key, nodeT(coeffT(cdata.v2k), true));
                }
            }
            else {
                if (key.level()<initial_level) {
                    coeffs.replace(key, nodeT(coeffT(), true));
                }
                else {
                    coeffs.replace(key, nodeT(coeffT(cdata.vk), false));
                }
            }
        }
        if (key.level() < initial_level) {
            for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                insert_zero_down_to_initial_level(kit.key());
            }
        }

    }

    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::print_tree(Level maxlevel) const {
        if (world.rank() == 0) do_print_tree(cdata.key0, maxlevel);
        world.gop.fence();
        if (world.rank() == 0) std::cout.flush();
        world.gop.fence();
    }


    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::do_print_tree(const keyT& key, Level maxlevel) const {
        typename dcT::const_iterator it = coeffs.find(key).get();
        if (it == coeffs.end()) {
            //MADNESS_EXCEPTION("FunctionImpl: do_print_tree: null node pointer",0);
            for (int i=0; i<key.level(); ++i) std::cout << "  ";
            std::cout << key << "  missing --> " << coeffs.owner(key) << "\n";
        }
        else {
            const nodeT& node = it->second;
            for (int i=0; i<key.level(); ++i) std::cout << "  ";
            std::cout << key << "  " << node << " --> " << coeffs.owner(key) << "\n";
            if (key.level() < maxlevel  &&  node.has_children()) {
                for (KeyChildIterator<NDIM> kit(key); kit; ++kit) {
                    do_print_tree(kit.key(),maxlevel);
                }
            }
        }
    }

    template <typename T, std::size_t NDIM>
    Tensor<T> FunctionImpl<T,NDIM>::project(const keyT& key) const {
        PROFILE_MEMBER_FUNC(FunctionImpl);

        if (not functor) MADNESS_EXCEPTION("FunctionImpl: project: confusion about function?",0);

        // if functor provides coeffs directly, awesome; otherwise use compute by yourself
        if (functor->provides_coeff()) return copy(functor->coeff(key));

        MADNESS_ASSERT(cdata.npt == cdata.k); // only necessary due to use of fast transform
        tensorT fval(cdata.vq,false); // this will be the returned result
        tensorT work(cdata.vk,false); // initially evaluate the function in here
        tensorT workq(cdata.vq,false); // initially evaluate the function in here

        madness::fcube(key,*functor,cdata.quad_x,work);

        work.scale(sqrt(FunctionDefaults<NDIM>::get_cell_volume()*pow(0.5,double(NDIM*key.level()))));
        //return transform(work,cdata.quad_phiw);
        return fast_transform(work,cdata.quad_phiw,fval,workq);
    }

    template <typename T, std::size_t NDIM>
    Void FunctionImpl<T,NDIM>::eval(const Vector<double,NDIM>& xin,
                                    const keyT& keyin,
                                    const typename Future<T>::remote_refT& ref) {

        PROFILE_MEMBER_FUNC(FunctionImpl);
        // This is ugly.  We must figure out a clean way to use
        // owner computes rule from the container.
        Vector<double,NDIM> x = xin;
        keyT key = keyin;
        Vector<Translation,NDIM> l = key.translation();
        ProcessID me = world.rank();
        while (1) {
            ProcessID owner = coeffs.owner(key);
            if (owner != me) {
                PROFILE_BLOCK(eval_send);
                woT::task(owner, &implT::eval, x, key, ref, TaskAttributes::hipri());
                return None;
            }
            else {
                typename dcT::futureT fut = coeffs.find(key);
                typename dcT::iterator it = fut.get();
                nodeT& node = it->second;
                if (node.has_coeff()) {
                    Future<T>(ref).set(eval_cube(key.level(), x, copy(node.coeff())));
                    return None;
                }
                else {
                    for (std::size_t i=0; i<NDIM; ++i) {
                        double xi = x[i]*2.0;
                        int li = int(xi);
                        if (li == 2) li = 1;
                        x[i] = xi - li;
                        l[i] = 2*l[i] + li;
                    }
                    key = keyT(key.level()+1,l);
                }
            }
        }
        //MADNESS_EXCEPTION("should not be here",0);
    }


    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::tnorm(const tensorT& t, double* lo, double* hi) const {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        // Chosen approach looks stupid but it is more accurate
        // than the simple approach of summing everything and
        // subtracting off the low-order stuff to get the high
        // order (assuming the high-order stuff is small relative
        // to the low-order)
        tensorT work = copy(t);
        tensorT tlo = work(cdata.sh);
        *lo = tlo.normf();
        tlo.fill(0.0);
        *hi = work.normf();
    }

    template <typename T, std::size_t NDIM>
    void FunctionImpl<T,NDIM>::phi_for_mul(Level np, Translation lp, Level nc, Translation lc, Tensor<double>& phi) const {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        double p[200];
        double scale = pow(2.0,double(np-nc));
        for (int mu=0; mu<cdata.npt; ++mu) {
            double xmu = scale*(cdata.quad_x(mu)+lc) - lp;
            MADNESS_ASSERT(xmu>-1e-15 && xmu<(1+1e-15));
            legendre_scaling_functions(xmu,cdata.k,p);
            for (int i=0; i<k; ++i) phi(i,mu) = p[i];
        }
        phi.scale(pow(2.0,0.5*np));
    }

    template <typename T, std::size_t NDIM>
    const Tensor<T> FunctionImpl<T,NDIM>::parent_to_child(const coeffT& s, const keyT& parent, const keyT& child) const {
        PROFILE_MEMBER_FUNC(FunctionImpl);
        // An invalid parent/child means that they are out of the box
        // and it is the responsibility of the caller to worry about that
        // ... most likely the coefficients (s) are zero to reflect
        // zero B.C. so returning s makes handling this easy.
        if (parent == child || parent.is_invalid() || child.is_invalid()) return s;

        coeffT result = fcube_for_mul<T>(child, parent, s);
        result.scale(sqrt(FunctionDefaults<NDIM>::get_cell_volume()*pow(0.5,double(NDIM*child.level()))));
        result = transform(result,cdata.quad_phiw);

        return result;
    }



    template <typename T, std::size_t NDIM>
    Future< Tensor<T> > FunctionImpl<T,NDIM>::compress_spawn(const Key<NDIM>& key,
            bool nonstandard, bool keepleaves, bool redundant) {
        if (!coeffs.probe(key)) print("missing node",key);
        MADNESS_ASSERT(coeffs.probe(key));

        // get fetches remote data (here actually local)
        nodeT& node = coeffs.find(key).get()->second;
        if (node.has_children()) {
            std::vector< Future<coeffT > > v = future_vector_factory<coeffT >(1<<NDIM);
            int i=0;
            for (KeyChildIterator<NDIM> kit(key); kit; ++kit,++i) {
                PROFILE_BLOCK(compress_send);
                // readily available
                v[i] = woT::task(coeffs.owner(kit.key()), &implT::compress_spawn, kit.key(),
                        nonstandard, keepleaves, redundant, TaskAttributes::hipri());
            }
            //if (redundant) return woT::task(world.rank(),&implT::make_redundant_op, key, v);
            return woT::task(world.rank(),&implT::compress_op, key, v, nonstandard, redundant);
        }
        else {
            Future<coeffT > result(node.coeff());
            if (!keepleaves) node.clear_coeff();
            return result;
        }
    }

    template <std::size_t NDIM>
    void FunctionDefaults<NDIM>::set_defaults(World& world) {
        k = 6;
        thresh = 1e-4;
        initial_level = 2;
        max_refine_level = 30;
        truncate_mode = 0;
        refine = true;
        autorefine = true;
        debug = false;
        truncate_on_project = true;
        apply_randomize = false;
        project_randomize = false;
        bc = BoundaryConditions<NDIM>(BC_FREE);
        tt = TT_FULL;
        cell = Tensor<double>(NDIM,2);
        cell(_,1) = 1.0;
        recompute_cell_info();

        //pmap = std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >(new WorldDCDefaultPmap< Key<NDIM> >(world));
        //pmap = std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >(new madness::LevelPmap< Key<NDIM> >(world));
        pmap = std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >(new SimplePmap< Key<NDIM> >(world));
    }

    template <typename T, std::size_t NDIM>
    const FunctionCommonData<T,NDIM>* FunctionCommonData<T,NDIM>::data[MAXK] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    template <std::size_t NDIM> int FunctionDefaults<NDIM>::k;
    template <std::size_t NDIM> double FunctionDefaults<NDIM>::thresh;
    template <std::size_t NDIM> int FunctionDefaults<NDIM>::initial_level;
    template <std::size_t NDIM> int FunctionDefaults<NDIM>::max_refine_level;
    template <std::size_t NDIM> int FunctionDefaults<NDIM>::truncate_mode;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::refine;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::autorefine;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::debug;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::truncate_on_project;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::apply_randomize;
    template <std::size_t NDIM> bool FunctionDefaults<NDIM>::project_randomize;
    template <std::size_t NDIM> BoundaryConditions<NDIM> FunctionDefaults<NDIM>::bc;
    template <std::size_t NDIM> TensorType FunctionDefaults<NDIM>::tt;
    template <std::size_t NDIM> Tensor<double> FunctionDefaults<NDIM>::cell;
    template <std::size_t NDIM> Tensor<double> FunctionDefaults<NDIM>::cell_width;
    template <std::size_t NDIM> Tensor<double> FunctionDefaults<NDIM>::rcell_width;
    template <std::size_t NDIM> double FunctionDefaults<NDIM>::cell_volume;
    template <std::size_t NDIM> double FunctionDefaults<NDIM>::cell_min_width;
    template <std::size_t NDIM> std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > > FunctionDefaults<NDIM>::pmap;

    template <std::size_t NDIM> std::vector< Key<NDIM> > Displacements<NDIM>::disp;
    template <std::size_t NDIM> std::vector< Key<NDIM> > Displacements<NDIM>::disp_periodicsum[64];

}

#endif // MADNESS_MRA_MRAIMPL_H__INCLUDED
