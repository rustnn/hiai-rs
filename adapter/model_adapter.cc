/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * CANN Adapter Layer - Model Build Implementation
 *
 * Wraps ge::Model and hiai::HiaiIrBuild.
 */

#include "model_adapter.h"
#include "adapter_internal.h"

#include <string>

namespace ddk {
extern "C" {

/* ── Model lifecycle ────────────────────────────────────────────────── */

CannModelHandle cann_model_create() {
    CANN_TRY
    return new CannModelImpl();
    CANN_CATCH_HANDLE
}

CannModelHandle cann_model_create_with_name(const char* name) {
    CANN_TRY
    if (!name) return nullptr;
    return new CannModelImpl(std::string(name));
    CANN_CATCH_HANDLE
}

void cann_model_destroy(CannModelHandle model) {
    delete model;
}

/* ── Model / Graph ──────────────────────────────────────────────────── */

CannStatus cann_model_set_graph(CannModelHandle model, CannGraphHandle graph) {
    CANN_TRY
    if (!model || !graph) return kInvalidPtr;
    model->model.SetGraph(graph->graph);
    return kSuccess;
    CANN_CATCH_STATUS
}

CannGraphHandle cann_model_get_graph(CannModelHandle model) {
    CANN_TRY
    if (!model) return nullptr;
    ge::Graph g = model->model.GetGraph();
    auto* impl = new CannGraphImpl();
    impl->graph = g;
    return reinterpret_cast<CannGraphHandle>(impl);
    CANN_CATCH_HANDLE
}

/* ── Build Options ──────────────────────────────────────────────────── */

CannBuildOptionsHandle cann_build_options_create(void) {
    CANN_TRY
    return new CannBuildOptsImpl();
    CANN_CATCH_HANDLE
}

void cann_build_options_destroy(CannBuildOptionsHandle options) {
    delete options;
}

CannStatus cann_build_options_set_mode(CannBuildOptionsHandle options, int32_t mode) {
    CANN_TRY
    if (!options) return kInvalidPtr;
    options->options.mode = (mode == 0) ? hiai::AUTO : hiai::CUSTOM;
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_weight_data_type(CannBuildOptionsHandle options,
                                                      int32_t weight_dtype) {
    CANN_TRY
    if (!options) return kInvalidPtr;
    options->options.weightDataType = (weight_dtype == 1)
        ? hiai::FP16 : hiai::FP32;
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_device_order(CannBuildOptionsHandle options,
                                                  const int32_t* devices,
                                                  int32_t device_count) {
    CANN_TRY
    if (!options || !devices || device_count <= 0) return kInvalidPara;
    options->options.modelDeviceOrder.clear();
    for (int32_t i = 0; i < device_count; ++i) {
        options->options.modelDeviceOrder.push_back(
            static_cast<hiai::ExecuteDevice>(devices[i]));
    }
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_input_shapes(CannBuildOptionsHandle options,
                                                  const int64_t* const* shapes,
                                                  const int32_t* shape_counts,
                                                  int32_t num_inputs) {
    CANN_TRY
    if (!options || !shapes || !shape_counts || num_inputs <= 0)
        return kInvalidPara;
    options->options.inputShapes.clear();
    for (int32_t i = 0; i < num_inputs; ++i) {
        if (!shapes[i]) return kInvalidPtr;
        std::vector<int64_t> s(shapes[i], shapes[i] + shape_counts[i]);
        options->options.inputShapes.push_back(s);
    }
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_precision_mode(CannBuildOptionsHandle options,
                                                    int32_t precision_mode) {
    CANN_TRY
    if (!options) return kInvalidPtr;
    /* PrecisionMode is set via AiModelDescription, not BuildOptions. */
    (void)precision_mode;
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_quantize_config(CannBuildOptionsHandle options,
                                                    const char* config) {
    CANN_TRY
    if (!options || !config) return kInvalidPtr;
    options->options.quantizeConfig = std::string(config);
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_options_set_tuning_strategy(CannBuildOptionsHandle options,
                                                    int32_t strategy) {
    CANN_TRY
    if (!options) return kInvalidPtr;
    options->options.tuningStrategy = static_cast<hiai::TuningStrategy>(strategy);
    return kSuccess;
    CANN_CATCH_STATUS
}

/* ── Model Building ──────────────────────────────────────────────────── */

CannHiaiIrBuildHandle cann_hiai_ir_build_create() {
    CANN_TRY
    auto* impl = new CannHiaiIrBuildImpl();
    return reinterpret_cast<CannHiaiIrBuildHandle>(impl);
    CANN_CATCH_HANDLE
}

CannStatus cann_hiai_ir_build_destroy(CannHiaiIrBuildHandle build) {
    if (!build) return kInvalidPtr;
    delete reinterpret_cast<CannHiaiIrBuildImpl*>(build);
    return kSuccess;
}

CannStatus cann_model_create_buff(CannHiaiIrBuildHandle build,
                                   CannModelHandle model,
                                   CannModelBuffer* output_buffer,
                                   uint32_t custom_size) {
    CANN_TRY
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    bool success = builder->CreateModelBuff(model->model, outputData, custom_size);

    if (!success) return kFailed;

    output_buffer->data = outputData.data;
    output_buffer->length = outputData.length;
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_model_create_buff_default(CannHiaiIrBuildHandle build,
                                           CannModelHandle model,
                                           CannModelBuffer* output_buffer) {
    CANN_TRY
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    bool success = builder->CreateModelBuff(model->model, outputData);

    if (!success) return kFailed;

    output_buffer->data = outputData.data;
    output_buffer->length = outputData.length;
    return kSuccess;
    CANN_CATCH_STATUS
}

CannStatus cann_build_model(CannHiaiIrBuildHandle build,
                               CannModelHandle model,
                               CannBuildOptionsHandle options,
                               CannModelBuffer* output_buffer) {
    CANN_TRY
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    outputData.data = output_buffer->data;
    outputData.length = output_buffer->length;
    bool success = false;

    if (options) {
        success = builder->BuildIRModel(model->model, outputData, options->options);
    } else {
        success = builder->BuildIRModel(model->model, outputData);
    }

    if (!success) return kFailed;
    output_buffer->length = outputData.length;
    output_buffer->data = outputData.data;

    return kSuccess;
    CANN_CATCH_STATUS
}

void cann_model_buffer_destroy(CannHiaiIrBuildHandle build, CannModelBuffer* buffer) {
    CANN_TRY
    if (!build || !buffer) return;
    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData data;
    data.data = buffer->data;
    data.length = buffer->length;
    builder->ReleaseModelBuff(data);
    buffer->data = nullptr;
    buffer->length = 0;
    CANN_CATCH_VOID
}
}  // extern "C"
}  // namespace ddk
