/* stream_buffer.h -- Incremental UTF-8 and textual stop marker filtering.
 *
 * TextBuffer collects bytes from llama token pieces. push() returns only full
 * UTF-8 characters which cannot start a known stop marker. finish() flushes
 * remaining text, replacing malformed or incomplete UTF-8 with U+FFFD.
 * A complete marker terminates the stream and is never returned to the caller.
 */
#ifndef TCLLLAMA_STREAM_BUFFER_H
#define TCLLLAMA_STREAM_BUFFER_H

#include <string>
#include <cstring>

class TextBuffer {
    std::string pending;
    bool stopped_ = false;
    bool filter_markers_ = true;
    static const char *const *markers() {
        static const char *const values[] = {
            "<end_of_turn>", "<start_of_turn>", "<|im_end|>",
            "<|eot_id|>", "<|endoftext|>", NULL
        };
        return values;
    }
    std::string take(bool final) {
        size_t end = pending.size();
        bool marker = false;
        for (const char *const *m = markers(); filter_markers_ && *m; ++m) {
            size_t pos = pending.find(*m);
            if (pos != std::string::npos && pos < end) {
                end = pos;
                marker = true;
            }
        }
        if (filter_markers_ && !final && !marker) {
            for (const char *const *m = markers(); *m; ++m) {
                size_t length = std::strlen(*m);
                for (size_t n = 1; n < length && n <= pending.size(); ++n) {
                    if (pending.compare(pending.size() - n, n, *m, n) == 0 &&
                        pending.size() - n < end) end = pending.size() - n;
                }
            }
        }
        std::string result;
        size_t i = 0;
        while (i < end) {
            unsigned char c = (unsigned char)pending[i];
            size_t count = c < 0x80 ? 1 : (c >= 0xc2 && c <= 0xdf ? 2 :
                (c >= 0xe0 && c <= 0xef ? 3 : (c >= 0xf0 && c <= 0xf4 ? 4 : 0)));
            if (count > 1 && i + count > end && !final && !marker) break;
            bool valid = count && i + count <= end;
            if (valid && count > 1) {
                for (size_t j = 1; j < count; ++j)
                    if (((unsigned char)pending[i + j] & 0xc0) != 0x80) valid = false;
                unsigned char second = (unsigned char)pending[i + 1];
                if ((c == 0xe0 && second < 0xa0) || (c == 0xed && second >= 0xa0) ||
                    (c == 0xf0 && second < 0x90) || (c == 0xf4 && second >= 0x90)) valid = false;
            }
            if (valid) {
                result.append(pending, i, count);
                i += count;
            } else {
                result.append("\xef\xbf\xbd", 3);
                ++i;
            }
        }
        pending.erase(0, i);
        if (marker) {
            pending.clear();
            stopped_ = true;
        }
        return result;
    }
public:
    explicit TextBuffer(bool filter_markers = true) : filter_markers_(filter_markers) {}
    std::string push(const char *bytes, size_t length) {
        if (stopped_) return std::string();
        pending.append(bytes, length);
        return take(false);
    }
    std::string finish() { return stopped_ ? std::string() : take(true); }
    bool stopped() const { return stopped_; }
};
#endif
