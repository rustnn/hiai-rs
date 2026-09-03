/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * CANN Adapter Layer - AiContext Wrapper
 *
 * Wraps hiai::AiContext with a pure C interface.
 */

#ifndef CANN_CONTEXT_ADAPTER_H
#define CANN_CONTEXT_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Context lifecycle ───────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannContextHandle cann_context_create();
CANN_ADAPTER_EXPORT void              cann_context_destroy(CannContextHandle context);

/* ── Key-Value parameters ────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_context_set_para(CannContextHandle context,
                                   const char* key,
                                   const char* value);

/* Returns an owned copy of the parameter value (free with cann_string_free),
 * or nullptr on error. An empty string is returned for both an unset key and
 * an empty value; use cann_context_get_all_keys to test key presence. */
CANN_ADAPTER_EXPORT const char* cann_context_get_para(CannContextHandle context,
                                    const char* key);

CANN_ADAPTER_EXPORT CannStatus cann_context_add_para(CannContextHandle context,
                                   const char* key,
                                   const char* value);

CANN_ADAPTER_EXPORT CannStatus cann_context_del_para(CannContextHandle context,
                                   const char* key);

CANN_ADAPTER_EXPORT CannStatus cann_context_clear_para(CannContextHandle context);

CANN_ADAPTER_EXPORT CannStatus cann_context_get_all_keys(CannContextHandle context,
                                       char** keys,
                                       int32_t max_keys,
                                       int32_t* out_key_count);

/* ── Shared string ownership ──────────────────────────────────────────── */

/* Frees a NUL-terminated string returned by any cann_* function that
 * allocates a copy for the caller (cann_context_get_para,
 * cann_context_get_all_keys, cann_model_desc_get_name,
 * cann_model_manager_get_version, cann_operator_get_name/type).
 * Passing nullptr is a no-op. */
CANN_ADAPTER_EXPORT void cann_string_free(const char* str);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_CONTEXT_ADAPTER_H */
