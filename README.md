# PET Simulation Project

## Part 1 - Forward Projection / Radon Transform Sinogram Model (ideal)

This is meant to be a noiseless sinogram, to later compare with one determined from simulated annihilation events. It's not perfect but it's a start.

I tried to do something similar to Andrew Reader, who has a great youtube series on this kind of thing. You can find it [here](https://www.youtube.com/playlist?list=PL557uxcMh3xzkLucqdvHgwU4zlheu_CKI).

Here is a gif of some of my results. Please enjoy. This is a WIP! 

[OUTDATED! UPDATED VERSION COMING SOON!]![gif sinogram](https://github.com/user-attachments/assets/3b26581f-8176-4361-9fd4-7cd824a435cb)


*Left: Random Emitters | Middle: (a few) Forward Projection Scan Lines for Integrating | Right: Sinogram*

UPDATE APRIL 9 2026
- Started preparing for FBP integration with looking at video series by Andrew Reader (see TODO)

UPDATE APRIL 8 2026

- Changed some names
- Changed out of bounds area colour to gray (emitters outside this region still not coulored over, but not used either)
- Updated sinogram to keep track of true counts, instead of maxing out at 255, so that the final image is remapped from [0, max_count] -> [0, 255], only iff max_count > 255. 
  - Not sure if best. Will have to implement FBP
- Increased IMAGE_SIZE, tested some higher counts
  - LAST TEST: ./build/PET 500 10, IMAGE_SIZE 1024
  - Better IMAGE_SIZE 500 because frame skipping not implemented (waiting each IMAGE_SIZE of pixels rows drawn of sinogram, so animation is quite slow for high counts)

TODO

- Implement FPB to test your sinogram

'''

    // basic steps from Andrew Reader, equivalent to 2D transform method
    // (take advantage of linearity, use 1D transforms )
    // projecting = vector(IMAGE_SIZE^2)
    // angle =0
    // angle_increment = increment
    // for (auto& row:img)
    //     1D FT row
    //     row.RampFilter (multiply by abs(index), middle is zero)
    //     row.inverseFFT
    //     set_projecting(projecting, row)
    //     angle+=increment
    //     back_project(projecting, angle, image)
'''
    
- Add axes
- Colour over OOB emitters
- Other quality of life checks
  - understanding the even circle centers (2x2 or 4x4 for example), should be okay but just to be sure
- arbitrary input images

"Utilized C++ and OpenCV to animate the Forward Projection or Radon Transform of a noiseless sinogram based on random emitter locations. This will eventually be compared with a noisy sinogram from simulated annihilation events, in the context of their reconstruction via Filtered Back Projection."

---

## Part 2 - Filtered Backprojection Method

## Part 3 - Annihilation Simulation

