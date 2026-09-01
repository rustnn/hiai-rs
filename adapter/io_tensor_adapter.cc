/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * CANN Adapter Layer - IOTensor Implementation
 */

#include "io_tensor_adapter.h"
#include "adapter_internal.h"

#include <cstring>
#include <vector>

#include "graph/types.h"

namespace ddk {
extern "C" {

/* ── Internal helpers: value mapping ───────────────────────────────────── */

static ge::DataType ToGeDataType(CannDataType t) {
    return static_cast<ge::DataType>(t);
}

static CannDataType FromGeDataType(ge::DataType t) {
    return static_cast<CannDataType>(t);
}

static ge::Format ToGeFormat(CannFormat f) {
    return static_cast<ge::Format>(f);
}

static CannFormat FromGeFormat(ge::Format f) {
    return static_cast<CannFormat>(f);
}

static hiai::HIAI_DataType ToHiaiDataType(CannDataType t) {
    switch (t) {
        case CANN_DT_UINT8:   return hiai::HIAI_DATATYPE_UINT8;
        case CANN_DT_FLOAT:   return hiai::HIAI_DATATYPE_FLOAT32;
        case CANN_DT_FLOAT16: return hiai::HIAI_DATATYPE_FLOAT16;
        case CANN_DT_INT32:   return hiai::HIAI_DATATYPE_INT32;
        case CANN_DT_INT8:    return hiai::HIAI_DATATYPE_INT8;
        case CANN_DT_INT16:   return hiai::HIAI_DATATYPE_INT16;
        case CANN_DT_BOOL:    return hiai::HIAI_DATATYPE_BOOL;
        case CANN_DT_INT64:   return hiai::HIAI_DATATYPE_INT64;
        case CANN_DT_UINT32:  return hiai::HIAI_DATATYPE_UINT32;
        case CANN_DT_DOUBLE:  return hiai::HIAI_DATATYPE_DOUBLE;
        default:              return hiai::HIAI_DATATYPE_FLOAT32;
    }
}



/* ── Tensor lifecycle ──────────────────────────────────────────────────── */

CannIOTensorHandle cann_io_tensor_create() {
    CANN_TRY
    auto* t = new CannIOTensorImpl();
    return reinterpret_cast<CannIOTensorHandle>(t);
    CANN_CATCH_HANDLE
}


void cann_io_tensor_destroy(CannIOTensorHandle tensor) {
    delete reinterpret_cast<CannIOTensorImpl*>(tensor);
}

/* ── Tensor data access ────────────────────────────────────────────────── */

CannStatus cann_io_tensor_set_data(CannIOTensorHandle tensor,
                                 const void* data,
                                 uint32_t size) {
    CANN_TRY
    if (!tensor || !data) return kInvalidPtr;
    auto* t = reinterpret_cast<CannIOTensorImpl*>(tensor);
    void* buf = t->tensor->GetBuffer();
    if (!buf) return kFailed;
    uint32_t bufSize = t->tensor->GetSize();
    uint32_t copySize = (size < bufSize) ? size : bufSize;
    std::memcpy(buf, data, copySize);
    return kSuccess;
    CANN_CATCH_STATUS
}

void* cann_io_tensor_get_buffer(CannIOTensorHandle tensor) {
    CANN_TRY
    if (!tensor) return nullptr;
    return reinterpret_cast<CannIOTensorImpl*>(tensor)->tensor->GetBuffer();
    CANN_CATCH_HANDLE
}

uint32_t cann_io_tensor_get_size(CannIOTensorHandle tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<CannIOTensorImpl*>(tensor)->tensor->GetSize();
}



/* ── Tensor Init ──────────────────────────────────────────────────────── */

CannStatus cann_io_tensor_init(CannIOTensorHandle tensor,
                              CannIOTensorDimensionHandle dim,
                              CannDataType dtype) {
    CANN_TRY
    if (!tensor || !dim) return kInvalidPtr;
    auto* t = reinterpret_cast<CannIOTensorImpl*>(tensor);
    hiai::AIStatus ret = t->tensor->Init(&dim->dim, ToHiaiDataType(dtype));
    if (ret == hiai::AI_SUCCESS) {
        t->dim = dim->dim;
    }
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
    CANN_CATCH_STATUS
}

CannStatus cann_io_tensor_init_with_data(CannIOTensorHandle tensor,
                                        const void* data,
                                        CannIOTensorDimensionHandle dim,
                                        CannDataType dtype) {
    CANN_TRY
    if (!tensor || !data || !dim) return kInvalidPtr;
    auto* t = reinterpret_cast<CannIOTensorImpl*>(tensor);
    hiai::AIStatus ret = t->tensor->Init(data, &dim->dim, ToHiaiDataType(dtype));
    if (ret == hiai::AI_SUCCESS) {
        t->dim = dim->dim;
    }
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
    CANN_CATCH_STATUS
}

/* ── Tensor Dimension access ──────────────────────────────────────────── */

CannStatus cann_io_tensor_set_tensor_dimension(CannIOTensorHandle tensor,
                                              CannIOTensorDimensionHandle dim) {
    CANN_TRY
    if (!tensor || !dim) return kInvalidPtr;
    auto* t = reinterpret_cast<CannIOTensorImpl*>(tensor);
    hiai::AIStatus ret = t->tensor->SetTensorDimension(&dim->dim);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
    CANN_CATCH_STATUS
}

CannIOTensorDimensionHandle cann_io_tensor_get_tensor_dimension(CannIOTensorHandle tensor) {
    CANN_TRY
    if (!tensor) return nullptr;
    auto* t = reinterpret_cast<CannIOTensorImpl*>(tensor);
    return new CannIOTensorDimImpl(t->tensor->GetTensorDimension());
    CANN_CATCH_HANDLE
}

/* ── TensorDimension lifecycle ──────────────────────────────────────────── */

CannIOTensorDimensionHandle cann_io_tensor_dim_create(uint32_t n, uint32_t c,
                                                   uint32_t h, uint32_t w) {
    CANN_TRY
    return new CannIOTensorDimImpl(n, c, h, w);
    CANN_CATCH_HANDLE
}

CannIOTensorDimensionHandle cann_io_tensor_dim_create_nd(const uint32_t* dims,
                                                      int32_t dim_count) {
    CANN_TRY
    if (!dims || dim_count <= 0) return nullptr;
    std::vector<uint32_t> v(dims, dims + dim_count);
    return new CannIOTensorDimImpl(v);
    CANN_CATCH_HANDLE
}

void cann_io_tensor_dim_destroy(CannIOTensorDimensionHandle dim) {
    delete dim;
}

void cann_io_tensor_dim_set_number(CannIOTensorDimensionHandle dim, uint32_t n) {
    CANN_TRY
    if (dim) dim->dim.SetNumber(n);
    CANN_CATCH_VOID
}

uint32_t cann_io_tensor_dim_get_number(CannIOTensorDimensionHandle dim) {
    return dim ? dim->dim.GetNumber() : 0;
}

void cann_io_tensor_dim_set_channel(CannIOTensorDimensionHandle dim, uint32_t c) {
    CANN_TRY
    if (dim) dim->dim.SetChannel(c);
    CANN_CATCH_VOID
}

uint32_t cann_io_tensor_dim_get_channel(CannIOTensorDimensionHandle dim) {
    return dim ? dim->dim.GetChannel() : 0;
}

void cann_io_tensor_dim_set_height(CannIOTensorDimensionHandle dim, uint32_t h) {
    CANN_TRY
    if (dim) dim->dim.SetHeight(h);
    CANN_CATCH_VOID
}

uint32_t cann_io_tensor_dim_get_height(CannIOTensorDimensionHandle dim) {
    return dim ? dim->dim.GetHeight() : 0;
}

void cann_io_tensor_dim_set_width(CannIOTensorDimensionHandle dim, uint32_t w) {
    CANN_TRY
    if (dim) dim->dim.SetWidth(w);
    CANN_CATCH_VOID
}

uint32_t cann_io_tensor_dim_get_width(CannIOTensorDimensionHandle dim) {
    return dim ? dim->dim.GetWidth() : 0;
}

}  // extern "C"
}  // namespace ddk
