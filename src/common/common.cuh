#pragma once

namespace bonsai {
  using Time = double;

  struct RateDef {
    Time start;
    Time end;
    // TODO: string curve

    RateDef(const Time start, const Time end) : start(start), end(end) {}
  };
}
