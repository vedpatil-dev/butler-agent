/* Fedora ships Khronos' OpenCL-ICD-Loader, not the legacy ocl-icd project, so <ocl_icd.h>
 * doesn't exist. XRT's xocl/api/icd/ocl_icd_bindings.h only needs _cl_icd_dispatch, which is
 * ABI-identical to Khronos' cl_icd_dispatch (<CL/cl_icd.h>, from the opencl-headers package). */
#pragma once
#include <CL/cl_icd.h>
typedef cl_icd_dispatch _cl_icd_dispatch;
