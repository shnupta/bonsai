#pragma once

namespace bonsai {
  using Time = double;
  Time SYSTEM_TIME = 0.0;

  // Specification of a Libor rate
  // TODO: Maybe move this into a different file?
  struct RateDef {
    Time start;
    Time end;
    // TODO: string curve

    RateDef(const Time start, const Time end) : start(start), end(end) {}
  };
}
