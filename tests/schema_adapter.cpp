/* schema_adapter.cpp -- Supported subset rejection checks, without a model. */
#include "../generic/schema_adapter.h"
#include <cassert>
#include <stdexcept>
int main() {
    const char *invalid[] = {
        "{", "[]", "false", "{}",
        R"({"type":"string","pattern":"abc"})",
        R"({"type":"integer","minimum":0})",
        R"({"type":"object","properties":{}})",
        R"({"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":1})",
        R"({"type":"string","enum":[true]})",
        R"({"type":"string","$ref":"#/x"})"
    };
    for (const char *s : invalid) {
        bool failed = false;
        try { ConvertJsonSchema(s); } catch (const std::invalid_argument &) { failed = true; }
        assert(failed);
    }
    assert(!ConvertJsonSchema(R"({"type":"string","enum":["a","b"]})").empty());
    assert(!ConvertJsonSchema(R"({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2})").empty());
    return 0;
}
