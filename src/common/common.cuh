#pragma once

namespace bonsai {
  using Time = double;

  // Specification of a Libor rate
  struct RateDef {
    Time start;
    Time end;
    // TODO: string curve

    RateDef(const Time start, const Time end) : start(start), end(end) {}
  };
}
