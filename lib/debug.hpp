#pragma once

#if defined(DEBUG)
#define debug_print(STR)               \
    do                                 \
    {                                  \
        std::cout << STR << std::endl; \
    } while (0)
#else
#define debug_print(STR)
#endif

#if defined(DEBUG)
#define debug_variable(X)                               \
    do                                                  \
    {                                                   \
        std::cout << (#X) << " = " << (X) << std::endl; \
    } while (0)
#else
#define debug_variable(STR)
#endif