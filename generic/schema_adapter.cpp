/* schema_adapter.cpp -- Validate supported keywords, then call llama.cpp's
 * JSON parser and GBNF converter. This does not implement conversion rules.
 */
#include "schema_adapter.h"
#include "json-schema-to-grammar.h"
#include <set>
#include <stdexcept>
using J = common_json;
static void CheckSchema(const J &s, int depth) {
    if (depth > 16 || !s.is_object()) throw std::invalid_argument("schema must be an object; nesting limit is 16");
    if (!s.contains("type") || !s.at("type").is_string()) throw std::invalid_argument("explicit string type is required");
    std::string type = s.at("type").get<std::string>();
    if (type != "object" && type != "array" && type != "string" && type != "integer" &&
        type != "number" && type != "boolean" && type != "null") throw std::invalid_argument("unsupported type");
    std::set<std::string> allowed = {"type", "title", "description"};
    bool literal = s.contains("enum") || s.contains("const");
    if (literal) {
        if (s.contains("enum") && s.contains("const")) throw std::invalid_argument("enum and const cannot be combined");
        allowed.insert("enum"); allowed.insert("const");
        auto check = [&](const J &v) {
            bool valid = type == "string" ? v.is_string() : type == "boolean" ? v.is_boolean() :
                type == "null" ? v.is_null() : type == "integer" ? v.is_number_integer() :
                type == "number" ? v.is_number() : false;
            if (!valid) throw std::invalid_argument("enum/const values must match a primitive type");
        };
        if (s.contains("const")) check(s.at("const"));
        else {
            const J &a = s.at("enum");
            if (!a.is_array() || a.empty() || a.size() > 64) throw std::invalid_argument("enum requires 1..64 values");
            for (const auto &v : a) check(v);
        }
    } else if (type == "object") {
        allowed.insert("properties"); allowed.insert("required"); allowed.insert("additionalProperties");
        if (!s.contains("additionalProperties") || !s.at("additionalProperties").is_boolean() ||
            s.at("additionalProperties").get<bool>()) throw std::invalid_argument("objects require additionalProperties:false");
        if (!s.contains("properties") || !s.at("properties").is_object() || s.at("properties").size() > 64)
            throw std::invalid_argument("properties must be an object with at most 64 entries");
        for (const auto &[key, value] : s.at("properties").items()) CheckSchema(value, depth + 1);
        if (s.contains("required")) {
            if (!s.at("required").is_array()) throw std::invalid_argument("required must be an array");
            std::set<std::string> seen;
            for (const auto &v : s.at("required")) {
                if (!v.is_string()) throw std::invalid_argument("required entries must be strings");
                auto name = v.get<std::string>();
                if (!s.at("properties").contains(name) || !seen.insert(name).second)
                    throw std::invalid_argument("required entries must be unique declared properties");
            }
        }
    } else if (type == "array") {
        allowed.insert("items"); allowed.insert("minItems"); allowed.insert("maxItems");
        if (!s.contains("items")) throw std::invalid_argument("arrays require items schema");
        CheckSchema(s.at("items"), depth + 1);
        int lo = 0, hi = 64;
        for (const char *key : {"minItems", "maxItems"}) {
            if (s.contains(key)) {
                if (!s.at(key).is_number_integer()) throw std::invalid_argument("array limits must be integers");
                auto n = s.at(key).get<int64_t>();
                if (n < 0 || n > 64) throw std::invalid_argument("array limits must be 0..64");
                if (std::string(key) == "minItems") lo = (int)n; else hi = (int)n;
            }
        }
        if (lo > hi) throw std::invalid_argument("minItems exceeds maxItems");
    }
    for (const auto &[key, value] : s.items()) {
        if (!allowed.count(key)) throw std::invalid_argument("unsupported keyword: " + key);
        if ((key == "title" || key == "description") && !value.is_string()) throw std::invalid_argument("annotations must be strings");
    }
}
std::string ConvertJsonSchema(const std::string &text) {
    J schema;
    try { schema = J::parse(text); }
    catch (const std::exception &e) { throw std::invalid_argument(std::string("JSON syntax error: ") + e.what()); }
    try { CheckSchema(schema, 0); return json_schema_to_grammar(schema, true); }
    catch (const std::exception &e) { throw std::invalid_argument(std::string("JSON Schema conversion error: ") + e.what()); }
}
