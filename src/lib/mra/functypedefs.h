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
#ifndef MADNESS_MRA_FUNCTYPEDEFS_H__INCLUDED
#define MADNESS_MRA_FUNCTYPEDEFS_H__INCLUDED

/// \file mra/functypedefs.h
/// \brief Provides typedefs to hide use of templates and to increase interoperability

namespace madness {
    typedef Tensor<double> real_tensor;
    typedef Tensor<double_complex> complex_tensor;

    typedef Tensor<double> tensor_real;
    typedef Tensor<double_complex> tensor_complex;

    typedef Vector<double,1> coord_1d;
    typedef Vector<double,2> coord_2d;
    typedef Vector<double,3> coord_3d;
    typedef Vector<double,4> coord_4d;
    typedef Vector<double,5> coord_5d;
    typedef Vector<double,6> coord_6d;

    typedef std::vector<double> vector_real;
    typedef std::vector< std::complex<double> > vector_complex;

    typedef std::vector< Vector<double,1> > vector_coord_1d;
    typedef std::vector< Vector<double,2> > vector_coord_2d;
    typedef std::vector< Vector<double,3> > vector_coord_3d;
    typedef std::vector< Vector<double,4> > vector_coord_4d;
    typedef std::vector< Vector<double,5> > vector_coord_5d;
    typedef std::vector< Vector<double,6> > vector_coord_6d;

    typedef Function<double,3> real_function_3d;

    typedef std::vector<real_function_3d> vector_real_function_3d;

    typedef FunctionFactory<double,3> real_factory_3d;
    typedef std::shared_ptr< FunctionFunctorInterface<double,3> > real_functor_3d;

    typedef std::shared_ptr< WorldDCPmapInterface< Key<3> > > pmap_3d;

    typedef FunctionImpl<double,3> real_funcimpl_3d;

    typedef FunctionDefaults<3> function_defaults_3d;

}

#endif  // MADNESS_MRA_MRA_H__INCLUDED
