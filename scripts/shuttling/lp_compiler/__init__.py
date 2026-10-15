"""AOD shuttling compiler for the LP-code models of the animation.

- ``layout``: tiles (fibres) of the data and ancillary codes, instances from
  ``algebraic_qec.examples.surgery_instances``;
- ``rounds``: transport rounds of each factor of the check matrices;
- ``motion``: AOD move times and the axis routes of in-tile sheet shifts;
- ``routes``: whole-tile shuttles and their passing lanes;
- ``schedule``: per-bank job scheduling and cross-bank retiming;
- ``replay``: the compiler from rounds to timed atom moves and pulses;
- ``models``: surgery (fibre, packet, adapter), memory and bare LP models;
- ``checks``: independent checks of the measured matrices and the physics.
"""
