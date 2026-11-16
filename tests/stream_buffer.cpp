/* stream_buffer.cpp -- Deterministic tests of token piece buffering.
 * Build with: c++ -std=c++11 tests/stream_buffer.cpp -o /tmp/stream-buffer-test
 */
#include "../generic/stream_buffer.h"
#include <cassert>
#include <string>

int main() {
    TextBuffer utf;
    std::string text;
    text += utf.push("A\xf0\x9f", 3);
    assert(text == "A");
    text += utf.push("\x98\x80", 2);
    text += utf.finish();
    assert(text == "A\xf0\x9f\x98\x80");
    TextBuffer marker;
    text = marker.push("hello <|im_", 11);
    text += marker.push("end|>ignored", 12);
    text += marker.finish();
    assert(marker.stopped() && text == "hello ");
    TextBuffer partial;
    text = partial.push("visible <end", 12);
    text += partial.finish();
    assert(text == "visible <end");
    TextBuffer long_piece;
    std::string long_text(1000, 'x');
    text = long_piece.push(long_text.data(), long_text.size());
    text += long_piece.finish();
    assert(text == long_text);
    TextBuffer incomplete;
    text = incomplete.push("ok\xe2\x82", 4);
    text += incomplete.finish();
    assert(text == "ok\xef\xbf\xbd\xef\xbf\xbd");
    return 0;
}
