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


#ifndef MADNESS_MRA_MRA_H__INCLUDED
#define MADNESS_MRA_MRA_H__INCLUDED

/*!
  \file mra/mra.h
  \brief Main include file for MADNESS and defines \c Function interface

  \addtogroup mra

*/


#include <world/world.h>
#include <misc/misc.h>
#include <tensor/tensor.h>

//#define FUNCTION_INSTANTIATE_3
//#if !defined(HAVE_IBMBGP) || !defined(HAVE_IBMBGQ)
//#define FUNCTION_INSTANTIATE_1
//#define FUNCTION_INSTANTIATE_2
//#define FUNCTION_INSTANTIATE_4
//#define FUNCTION_INSTANTIATE_5
//#define FUNCTION_INSTANTIATE_6
//#endif

static const bool VERIFY_TREE = false; //true;


namespace madness {
    void startup(World& world, int argc, char** argv);
}

#include <mra/key.h>
#include <mra/twoscale.h>
#include <mra/legendre.h>
#include <mra/indexit.h>
#include <world/worlddc.h>
#include <mra/funcdefaults.h>
#include <mra/function_factory.h>

// some forward declarations
namespace madness {

    template<typename T, std::size_t NDIM>
    class FunctionImpl;

    template<typename T, std::size_t NDIM>
    class Function;

    template<typename T, std::size_t NDIM>
    class FunctionNode;

    template<typename T, std::size_t NDIM>
    class FunctionFactory;

    template<typename T, std::size_t NDIM>
    class FunctionFunctorInterface;
}


namespace madness {
    /// \ingroup mra
    /// \addtogroup function

    /// A multiresolution adaptive numerical function
    template <typename T, std::size_t NDIM>
    class Function {
        // We make all of the content of Function and FunctionImpl
        // public with the intent of avoiding the cumbersome forward
        // and friend declarations.  However, this open access should
        // not be abused.

    private:
        std::shared_ptr< FunctionImpl<T,NDIM> > impl;

    public:
        typedef FunctionImpl<T,NDIM> implT;
        typedef FunctionNode<T,NDIM> nodeT;
        typedef FunctionFactory<T,NDIM> factoryT;
        typedef Vector<double,NDIM> coordT; ///< Type of vector holding coordinates

        /// Asserts that the function is initialized
        inline void verify() const {
            MADNESS_ASSERT(impl);
        }

        /// Returns true if the function is initialized
        bool is_initialized() const {
            return impl.get();
        }

        /// Default constructor makes uninitialized function.  No communication.

        /// An unitialized function can only be assigned to.  Any other operation will throw.
        Function() : impl() {}


        /// Constructor from FunctionFactory provides named parameter idiom.  Possible non-blocking communication.
        Function(const factoryT& factory)
                : impl(new FunctionImpl<T,NDIM>(factory)) {
            PROFILE_MEMBER_FUNC(Function);
        }


        /// Copy constructor is \em shallow.  No communication, works in either basis.
        Function(const Function<T,NDIM>& f)
                : impl(f.impl) {
        }


        /// Assignment is \em shallow.  No communication, works in either basis.
        Function<T,NDIM>& operator=(const Function<T,NDIM>& f) {
            PROFILE_MEMBER_FUNC(Function);
            if (this != &f) impl = f.impl;
            return *this;
        }

        /// Destruction of any underlying implementation is deferred to next global fence.
        ~Function() {}

        /// Evaluates the function at a point in user coordinates.  Possible non-blocking comm.

        /// Only the invoking process will receive the result via the future
        /// though other processes may be involved in the evaluation.
        ///
        /// Throws if function is not initialized.
        Future<T> eval(const coordT& xuser) const {
            PROFILE_MEMBER_FUNC(Function);
            const double eps=1e-15;
            verify();
            MADNESS_ASSERT(!is_compressed());
            coordT xsim;
            user_to_sim(xuser,xsim);
            // If on the boundary, move the point just inside the
            // volume so that the evaluation logic does not fail
            for (std::size_t d=0; d<NDIM; ++d) {
                if (xsim[d] < -eps) {
                    MADNESS_EXCEPTION("eval: coordinate lower-bound error in dimension", d);
                }
                else if (xsim[d] < eps) {
                    xsim[d] = eps;
                }

                if (xsim[d] > 1.0+eps) {
                    MADNESS_EXCEPTION("eval: coordinate upper-bound error in dimension", d);
                }
                else if (xsim[d] > 1.0-eps) {
                    xsim[d] = 1.0-eps;
                }
            }

            Future<T> result;
            impl->eval(xsim, impl->key0(), result.remote_ref(impl->world));
            return result;
        }


        /// Evaluates the function at a point in user coordinates.  Collective operation.

        /// Throws if function is not initialized.
        ///
        /// This function calls eval, blocks until the result is
        /// available and then broadcasts the result to everyone.
        /// Therefore, if you are evaluating many points in parallel
        /// it is \em vastly less efficient than calling eval
        /// directly, saving the futures, and then forcing all of the
        /// results.
        T operator()(const coordT& xuser) const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            if (is_compressed()) reconstruct();
            T result;
            if (impl->world.rank() == 0) result = eval(xuser).get();
            impl->world.gop.broadcast(result);
            //impl->world.gop.fence();
            return result;
        }

        /// Evaluates the function at a point in user coordinates.  Collective operation.

        /// See "operator()(const coordT& xuser)" for more info
        T operator()(double x, double y=0, double z=0, double xx=0, double yy=0, double zz=0) const {
            coordT r;
            r[0] = x;
            if (NDIM>=2) r[1] = y;
            if (NDIM>=3) r[2] = z;
            if (NDIM>=4) r[3] = xx;
            if (NDIM>=5) r[4] = yy;
            if (NDIM>=6) r[5] = zz;
            return (*this)(r);
        }


        /// Verifies the tree data structure ... global sync implied
        void verify_tree() const {
            PROFILE_MEMBER_FUNC(Function);
            if (impl) impl->verify_tree();
        }


        /// Returns true if compressed, false otherwise.  No communication.

        /// If the function is not initialized, returns false.
        bool is_compressed() const {
            PROFILE_MEMBER_FUNC(Function);
            if (impl)
                return impl->is_compressed();
            else
                return false;
        }


        /// Returns the number of coefficients in the function ... collective global sum
        std::size_t size() const {
            PROFILE_MEMBER_FUNC(Function);
            if (!impl) return 0;
            return impl->size();
        }


        /// Returns value of truncation threshold.  No communication.
        double thresh() const {
            PROFILE_MEMBER_FUNC(Function);
            if (!impl) return 0.0;
            return impl->get_thresh();
        }


        /// Sets the vaule of the truncation threshold.  Optional global fence.

        /// A fence is required to ensure consistent global state.
        void set_thresh(double value, bool fence = true) {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            impl->set_thresh(value);
            if (fence) impl->world.gop.fence();
        }


        /// Returns the number of multiwavelets (k).  No communication.
        int k() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            return impl->get_k();
        }


        /// Returns a shared-pointer to the implementation
        const std::shared_ptr< FunctionImpl<T,NDIM> >& get_impl() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            return impl;
        }


        /// Returns the world
        World& world() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            return  impl->world;
        }


        /// Returns a shared pointer to the process map
        const std::shared_ptr< WorldDCPmapInterface< Key<NDIM> > >& get_pmap() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            return impl->get_pmap();
        }


        /// Returns the square of the norm of the local function ... no communication

        /// Works in either basis
        double norm2sq_local() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            return impl->norm2sq_local();
        }


        /// Returns the 2-norm of the function ... global sum ... works in either basis

        /// See comments for err() w.r.t. applying to many functions.
        double norm2() const {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            if (VERIFY_TREE) verify_tree();
            double local = impl->norm2sq_local();

            impl->world.gop.sum(local);
            impl->world.gop.fence();
            return sqrt(local);
        }


        /// Compresses the function, transforming into wavelet basis.  Possible non-blocking comm.

        /// By default fence=true meaning that this operation completes before returning,
        /// otherwise if fence=false it returns without fencing and the user must invoke
        /// world.gop.fence() to assure global completion before using the function
        /// for other purposes.
        ///
        /// Noop if already compressed or if not initialized.
        ///
        /// Since reconstruction/compression do not discard information we define them
        /// as const ... "logical constness" not "bitwise constness".
        const Function<T,NDIM>& compress(bool fence = true) const {
            PROFILE_MEMBER_FUNC(Function);
            if (!impl || is_compressed()) return *this;
            if (VERIFY_TREE) verify_tree();
            impl->world.gop.fence();
            const_cast<Function<T,NDIM>*>(this)->impl->compress(false, false, false, fence);
            return *this;
        }


        /// Compresses the function retaining scaling function coeffs.  Possible non-blocking comm.

        /// By default fence=true meaning that this operation completes before returning,
        /// otherwise if fence=false it returns without fencing and the user must invoke
        /// world.gop.fence() to assure global completion before using the function
        /// for other purposes.
        ///
        /// Noop if already compressed or if not initialized.
        void nonstandard(bool keepleaves, bool fence=true) {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            if (impl->is_nonstandard()) return;
            if (VERIFY_TREE) verify_tree();
            if (is_compressed()) reconstruct();
            impl->compress(true, keepleaves, false, fence);
        }


        /// Converts the function from nonstandard form to standard form.  Possible non-blocking comm.

        /// By default fence=true meaning that this operation completes before returning,
        /// otherwise if fence=false it returns without fencing and the user must invoke
        /// world.gop.fence() to assure global completion before using the function
        /// for other purposes.
        ///
        /// Must be already compressed.
        void standard(bool fence = true) {
            PROFILE_MEMBER_FUNC(Function);
            verify();
            MADNESS_ASSERT(is_compressed());
            impl->standard(fence);
            if (fence && VERIFY_TREE) verify_tree();
        }

        /// Reconstructs the function, transforming into scaling function basis.  Possible non-blocking comm.

        /// By default fence=true meaning that this operation completes before returning,
        /// otherwise if fence=false it returns without fencing and the user must invoke
        /// world.gop.fence() to assure global completion before using the function
        /// for other purposes.
        ///
        /// Noop if already reconstructed or if not initialized.
        ///
        /// Since reconstruction/compression do not discard information we define them
        /// as const ... "logical constness" not "bitwise constness".
        void reconstruct(bool fence = true) const {
            PROFILE_MEMBER_FUNC(Function);
            if (!impl || !is_compressed()) return;
            const_cast<Function<T,NDIM>*>(this)->impl->reconstruct(fence);
            if (fence && VERIFY_TREE) verify_tree(); // Must be after in case nonstandard
        }



        /// Clears the function as if constructed uninitialized.  Optional fence.

        /// Any underlying data will not be freed until the next global fence.
        void clear(bool fence = true) {
            PROFILE_MEMBER_FUNC(Function);
            if (impl) {
                World& world = impl->world;
                impl.reset();
                if (fence) world.gop.fence();
            }
        }

        /// Process 0 prints a summary of all nodes in the tree (collective)
        void print_tree() const {
            PROFILE_MEMBER_FUNC(Function);
            if (impl) impl->print_tree();
        }


        /// Returns local part of inner product ... throws if both not compressed
        template <typename R>
        TENSOR_RESULT_TYPE(T,R) inner_local(const Function<R,NDIM>& g) const {
            PROFILE_MEMBER_FUNC(Function);
            MADNESS_ASSERT(is_compressed());
            MADNESS_ASSERT(g.is_compressed());
            if (VERIFY_TREE) verify_tree();
            if (VERIFY_TREE) g.verify_tree();
            return impl->inner_local(*(g.get_impl()));
        }



        /// Returns the inner product

        /// Not efficient for computing multiple inner products
        /// @param[in]  g   Function, optionally on-demand
        template <typename R>
        TENSOR_RESULT_TYPE(T,R) inner(const Function<R,NDIM>& g) const {
            PROFILE_MEMBER_FUNC(Function);

            // fast return if possible
            if (not this->is_initialized()) return 0.0;
            if (not g.is_initialized()) return 0.0;

            // if this and g are the same, use norm2()
            if (this->get_impl()==g.get_impl()) {
                double norm=this->norm2();
                return norm*norm;
            }

            // do it case-by-case
            if (this->is_on_demand()) return g.inner_on_demand(*this);
            if (g.is_on_demand()) return this->inner_on_demand(g);

            if (VERIFY_TREE) verify_tree();
            if (VERIFY_TREE) g.verify_tree();

            // compression is more efficient for 3D
            if (NDIM==3) {
            	if (!is_compressed()) compress(false);
            	if (!g.is_compressed()) g.compress(false);
                impl->world.gop.fence();
           }

            if (this->is_compressed() and g.is_compressed()) {
            } else {
                if (not this->get_impl()->is_redundant()) this->get_impl()->make_redundant(false);
                if (not g.get_impl()->is_redundant()) g.get_impl()->make_redundant(false);
                impl->world.gop.fence();
            }


            TENSOR_RESULT_TYPE(T,R) local = impl->inner_local(*g.get_impl());
            impl->world.gop.sum(local);
            impl->world.gop.fence();

            if (this->get_impl()->is_redundant()) this->get_impl()->undo_redundant(false);
            if (g.get_impl()->is_redundant()) g.get_impl()->undo_redundant(false);
            impl->world.gop.fence();

            return local;
        }

        /// This is replaced with alpha*left + beta*right ...  private
        template <typename L, typename R>
        Function<T,NDIM>& gaxpy_oop(T alpha, const Function<L,NDIM>& left,
                                    T beta,  const Function<R,NDIM>& right, bool fence) {
            PROFILE_MEMBER_FUNC(Function);
            left.verify();
            right.verify();
            MADNESS_ASSERT(left.is_compressed() && right.is_compressed());
            if (VERIFY_TREE) left.verify_tree();
            if (VERIFY_TREE) right.verify_tree();
            impl.reset(new implT(*left.get_impl(), left.get_pmap(), false));
            impl->gaxpy(alpha,*left.get_impl(),beta,*right.get_impl(),fence);
            return *this;
        }

        /// Replace current FunctionImpl with provided new one
        void set_impl(const std::shared_ptr< FunctionImpl<T,NDIM> >& impl) {
            PROFILE_MEMBER_FUNC(Function);
            this->impl = impl;
        }

        /// Replace current FunctionImpl with a new one using the same parameters & map as f

        /// If zero is true the function is initialized to zero, otherwise it is empty
        template <typename R>
        void set_impl(const Function<R,NDIM>& f, bool zero = true) {
            impl = std::shared_ptr<implT>(new implT(*f.get_impl(), f.get_pmap(), zero));
        }
    };


    /// Sparse multiplication --- left and right must be reconstructed and if tol!=0 have tree of norms already created
    template <typename L, typename R,std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R),NDIM>
    mul_sparse(const Function<L,NDIM>& left, const Function<R,NDIM>& right, double tol, bool fence=true) {
        PROFILE_FUNC;
        left.verify();
        right.verify();
        MADNESS_ASSERT(!(left.is_compressed() || right.is_compressed()));
        if (VERIFY_TREE) left.verify_tree();
        if (VERIFY_TREE) right.verify_tree();

        Function<TENSOR_RESULT_TYPE(L,R),NDIM> result;
        result.set_impl(left, false);
        result.get_impl()->mulXX(left.get_impl().get(), right.get_impl().get(), tol, fence);
        return result;
    }

    /// Same as \c operator* but with optional fence and no automatic reconstruction
    template <typename L, typename R,std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R),NDIM>
    mul(const Function<L,NDIM>& left, const Function<R,NDIM>& right, bool fence=true) {
        return mul_sparse(left,right,0.0,fence);
    }

    /// Multiplies two functions with the new result being of type TensorResultType<L,R>

    /// Using operator notation forces a global fence after each operation but also
    /// enables us to automatically reconstruct the input functions as required.
    template <typename L, typename R, std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R), NDIM>
    operator*(const Function<L,NDIM>& left, const Function<R,NDIM>& right) {
        if (left.is_compressed())  left.reconstruct();
        if (right.is_compressed()) right.reconstruct();
        //MADNESS_ASSERT(not (left.is_on_demand() or right.is_on_demand()));
        return mul(left,right,true);
    }

    /// Returns new function alpha*left + beta*right optional fence and no automatic compression
    template <typename L, typename R,std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R),NDIM>
    gaxpy_oop(TENSOR_RESULT_TYPE(L,R) alpha, const Function<L,NDIM>& left,
              TENSOR_RESULT_TYPE(L,R) beta,  const Function<R,NDIM>& right, bool fence=true) {
        PROFILE_FUNC;
        Function<TENSOR_RESULT_TYPE(L,R),NDIM> result;
        return result.gaxpy_oop(alpha, left, beta, right, fence);
    }

    /// Same as \c operator+ but with optional fence and no automatic compression
    template <typename L, typename R,std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R),NDIM>
    add(const Function<L,NDIM>& left, const Function<R,NDIM>& right, bool fence=true) {
        return gaxpy_oop(TENSOR_RESULT_TYPE(L,R)(1.0), left,
                         TENSOR_RESULT_TYPE(L,R)(1.0), right, fence);
    }


    /// Adds two functions with the new result being of type TensorResultType<L,R>

    /// Using operator notation forces a global fence after each operation
    template <typename L, typename R, std::size_t NDIM>
    Function<TENSOR_RESULT_TYPE(L,R), NDIM>
    operator+(const Function<L,NDIM>& left, const Function<R,NDIM>& right) {
        if (VERIFY_TREE) left.verify_tree();
        if (VERIFY_TREE) right.verify_tree();

        // no compression for high-dimensional functions
        if (!left.is_compressed())  left.compress();
        if (!right.is_compressed()) right.compress();
        return add(left,right,true);
    }

    /// Computes the scalar/inner product between two functions

    /// In Maple this would be \c int(conjugate(f(x))*g(x),x=-infinity..infinity)
    template <typename T, typename R, std::size_t NDIM>
    TENSOR_RESULT_TYPE(T,R) inner(const Function<T,NDIM>& f, const Function<R,NDIM>& g) {
        PROFILE_FUNC;
        return f.inner(g);
    }


}

namespace madness {
    namespace archive {
        template <class T, std::size_t NDIM>
        struct ArchiveLoadImpl< ParallelInputArchive, Function<T,NDIM> > {
            static inline void load(const ParallelInputArchive& ar, Function<T,NDIM>& f) {
                f.load(*ar.get_world(), ar);
            }
        };

        template <class T, std::size_t NDIM>
        struct ArchiveStoreImpl< ParallelOutputArchive, Function<T,NDIM> > {
            static inline void store(const ParallelOutputArchive& ar, const Function<T,NDIM>& f) {
                f.store(ar);
            }
        };
    }


}
/* @} */

#include <mra/functypedefs.h>
#include <mra/mraimpl.h>

#endif // MADNESS_MRA_MRA_H__INCLUDED
