#ifndef __RPC_RESPONSE
#define __RPC_RESPONSE

#include <cereal/cereal.hpp>
#include "debug.hpp"

#include <torch/extension.h>

enum RPCResponseType { 
    Representative,
    Activation
};

struct rpc_response_t {

    RPCResponseType m_type;
    size_t m_num_elements;
    size_t m_offset;
    int m_label = -1;
    float m_weight = 0;

    rpc_response_t(RPCResponseType type, size_t num_elements, size_t offset) : m_type(type), m_num_elements(num_elements), m_offset(offset) { }
    rpc_response_t(RPCResponseType type, size_t num_elements, size_t offset, int label, float weight) : m_type(type), m_num_elements(num_elements), m_offset(offset), m_label(label), m_weight(weight) { }
    rpc_response_t() { }

    template<class Archive>
    void serialize(Archive& archive) {
        archive(m_type, m_num_elements, m_offset, m_label, m_weight);
    }
};

#endif
