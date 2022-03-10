# bonsai

General notes for hybrid (both `__host__` and `__device__`) data structures
is that they __can__ be dynamic in host code to allow for easy allocation and 
initialisation. Then once in device code their sizes are fixed, but you are
able to make modifications to the data stored. Some classes do have mixed in
`std::*` members, I'm still unsure if this will work out or if I'll have to add
some more of my own classes.
