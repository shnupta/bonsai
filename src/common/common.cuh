#pragma once

namespace bonsai {
  using Time = double;
  Time SYSTEM_TIME = 0.0;

  // Specification of a Libor rate
  struct RateDef {
    Time start;
    Time end;
    // TODO: string curve

    RateDef(const Time start, const Time end) : start(start), end(end) {}
  };
}
