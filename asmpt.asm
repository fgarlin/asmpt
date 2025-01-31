%define PCG32_INIT_STATE 0x853c49e6748fea9b
%define PCG32_INIT_SEQ   0xda3e39cb94b95bdb
%define IMAGE_WIDTH 640
%define IMAGE_HEIGHT 480
%define IMAGE_ARRAY_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3)
%define NUM_SAMPLES 1024

section .text
global _start

print_image:
    ; Write the header
    mov rax, 1                  ; write syscall
    mov rdi, 1                  ; stdout
    mov rsi, header             ; buffer address
    mov rdx, header_length      ; buffer size
    syscall
    ; Write the image itself
    mov rax, 1                  ; write syscall
    mov rdi, 1                  ; stdout
    mov rsi, image_array        ; buffer address
    mov rdx, IMAGE_ARRAY_SIZE   ; buffer size
    syscall

rand:
    ; Generate a random 32-bit integer and put it on eax
    ; Modifies: rcx, rax, rdi
    mov rcx, r15
    mov rax, r14
    shr rax, 18
    xor rax, r14
    shr rax, 27
    shr rcx, 59
    ror eax, cl
    ; Advance internal state
    mov rdi, 6364136223846793005
    imul r14, rdi
    mov rdi, r15
    or rdi, 1
    add r14, rdi
    ret

rand_float:
    ; Generates a random normalized [0,1) single-precision floating-point number
    ; Outputs to xmm0
    ; Modifies: xmm0, xmm1, eax
    call rand                   ; random 32-bit integer in eax
    ; Floating point conversion: https://fgarlin.com/posts/2025-01-17-gpu_rng/
    shr eax, 9
    or eax, 0x3F800000
    movd xmm0, eax
    movss xmm1, [float_1]
    subss xmm0, xmm1
    ret

vec_normalize:
    ; Normalize a vector in xmm0
    ; Modifies: xmm0, xmm1
    movaps xmm1, xmm0
    dpps xmm1, xmm1, 0xff       ; vector length broadcasted to all elements
    rsqrtps xmm1, xmm1
    mulps xmm0, xmm1
    ret

vec_cross:
    ; Cross product of two vectors (a x b)
    ; https://geometrian.com/resources/cross_product/
    ; Arguments:
    ; - xmm0: vector a
    ; - xmm1: vector b
    ; Returns:
    ; - xmm0: cross product
    ; Modifies: xmm0, xmm1, xmm2
    shufps xmm0, xmm0, 0xc9
    movaps xmm2, xmm0
    mulps xmm2, xmm1
    shufps xmm1, xmm1, 0xd2
    mulps xmm0, xmm1
    shufps xmm2, xmm2, 0xc9
    subps xmm0, xmm2
    ret

vec_reflect:
    ; Calculate the reflection direction for an incident vector
    ; Arguments:
    ; - xmm0: normal (must be normalized)
    ; - xmm4: ray direction (end point of the vector is the intersection point)
    ; Returns:
    ; - xmm4: reflected ray direction
    ; Modifies: xmm0, xmm1, xmm4
    ; r = d - 2 * dot(n, d) * n
    movaps xmm1, xmm0           ; xmm1 = n
    dpps xmm1, xmm4, 0xff       ; xmm1 = dot(n, d), broadcasted to all
    addps xmm1, xmm1            ; xmm1 = 2 * dot(n, d)
    mulps xmm1, xmm0            ; xmm1 = 2 * dot(n, d) * n
    subps xmm4, xmm1            ; xmm4 = r = d - 2 * dot(n, d) * n
    ret

sin:
    ; Arguments:
    ; - esi: angle (0 to 4095, where 0 is 0 and 4095 is 2*pi)
    ; Returns:
    ; - xmm0: sine of the angle as a 32-bit floating-point number
    ; Modifies: esi, edi, xmm0
    and esi, 4095               ; ensure angle is in correct range
    mov edi, esi
    and edi, 2048
    jnz sin_III
    mov edi, esi
    and edi, 1024
    jnz sin_II
    ; Angle is in the 1st quadrant (0-1023)
    jmp sin_ret
sin_II:
    ; Angle is in the 2nd quadrant (1024-2047)
    neg esi
    add esi, 2048
    jmp sin_ret
sin_III:
    mov edi, esi
    and edi, 1024
    jnz sin_IV
    ; Angle is in the 3rd quadrant (2048-3071)
    sub esi, 2048
    jmp sin_neg_ret
sin_IV:
    ; Angle is in the 4th quadrant (3072-4095)
    neg esi
    add esi, 4096
    jmp sin_neg_ret
sin_ret:
    movd xmm0, [sine + esi * 4]
    ret
sin_neg_ret:
    movd xmm0, [sine + esi * 4]
    xorps xmm0, [sign_bit]
    ret

cos:
    ; Same as sin, but with an offset of pi/2
    add esi, 1024
    call sin
    ret

rand_dir_cos_weighted_hemisphere:
    ; Return a random cosine-weighted direction in the local hemisphere.
    ; The returned vector is in world space.
    ; Arguments:
    ; - xmm0: u
    ; - xmm1: v
    ; - xmm2: n
    ; Returns:
    ; - xmm4: ray direction
    ; Modifies: xmm0, xmm1, xmm2, xmm3, xmm4, what rand modifies
    ; {
    ;   u * sin(theta) * cos(phi),
    ;   v * sin(theta) * sin(phi),
    ;   n * cos(theta)
    ; }
    movaps xmm3, xmm0           ; xmm3 = u
    movaps xmm4, xmm1           ; xmm4 = v

    call rand_float
    sqrtss xmm1, xmm0           ; xmm1 = cos(theta) = sqrt(rand)
    shufps xmm1, xmm1, 0        ; broadcast xmm1
    mulps xmm2, xmm1            ; xmm2 = n * cos(theta)

    movss xmm1, [float_1]       ; xmm1 = 1.0
    subss xmm1, xmm0            ; xmm1 = 1.0 - rand
    sqrtss xmm1, xmm1           ; xmm1 = sin(theta) = sqrt(1.0 - rand)
    shufps xmm1, xmm1, 0        ; broadcast xmm1
    mulps xmm3, xmm1            ; xmm3 = u * sin(theta)
    mulps xmm4, xmm1            ; xmm4 = v * sin(theta)

    call rand
    mov esi, eax
    call sin                    ; xmm0 = sin(phi)
    shufps xmm0, xmm0, 0        ; xmm0 broadcast to all
    mulps xmm4, xmm0            ; xmm4 = v * sin(phi)

    mov esi, eax
    call cos                    ; xmm0 = cos(phi)
    shufps xmm0, xmm0, 0        ; xmm0 broadcast to all
    mulps xmm3, xmm0            ; xmm3 = u * cos(phi)

    addps xmm4, xmm3            ; xmm4 = u + v
    addps xmm4, xmm2            ; xmm4 = u + v + n
    ret

coordinate_system:
    ; For a given vector in xmm0, complete the remaining vectors to create an
    ; orthonormal coordinate system in xmm1 and xmm2.
    ; Modifies: xmm0, xmm1, xmm2, xmm3, xmm4
    movss xmm4, [float_01]      ; xmm4 = 0.1
    ucomiss xmm0, xmm4
    jb other_major_axis         ; jump if n[0] < 0.1
    movups xmm1, [vec_up]       ; xmm1 = up
    jmp coord_common
other_major_axis:
    movups xmm1, [vec_up2]      ; xmm1 = up
coord_common:
    movaps xmm3, xmm0           ; xmm3 = n
    call vec_cross              ; xmm2 = u = n x up
    call vec_normalize
    movaps xmm4, xmm0           ; xmm4 = u
    movaps xmm0, xmm3           ; xmm0 = n
    movaps xmm1, xmm4           ; xmm1 = u
    call vec_cross              ; xmm2 = v = n x u
    call vec_normalize
    movaps xmm1, xmm0           ; xmm1 = v
    movaps xmm0, xmm4           ; xmm0 = u
    movaps xmm2, xmm3           ; xmm2 = n
    ret

intersect_sphere:
    ; Calculate the intersection between a ray and a sphere
    ; Arguments:
    ; - xmm0: sphere radius {r, r, r, r}
    ; - xmm1: sphere center
    ; - xmm4: ray direction
    ; - xmm5: ray origin
    ; Returns:
    ; - xmm0: Distance along ray where the intersection happened, or 0
    ; - ZF=1 means no intersection
    ; Modifies: xmm0, xmm1, xmm2, xmm3, eax
    movaps xmm2, xmm4           ; xmm2 = rd
    mulps xmm0, xmm0            ; xmm0 = r*r
    subps xmm1, xmm5            ; xmm1 = op = so - ro
    dpps xmm2, xmm1, 0xff       ; xmm2 = b = dot(op, rd)
    dpps xmm1, xmm1, 0xff       ; xmm1 = dot(op, op)

    movaps xmm3, xmm2           ; xmm3 = b
    mulps xmm3, xmm3            ; xmm3 = b*b
    subps xmm3, xmm1            ; xmm3 = b*b - dot(op, op)
    addps xmm3, xmm0            ; xmm3 = det = b*b - dot(op, op) + r*r

    xorps xmm0, xmm0            ; xmm0 = 0
    ucomiss xmm3, xmm0          ; compare det against 0
    jb .no_intersect            ; jump if det < 0

    movups xmm1, [float_epsilon]; xmm1 = epsilon
    sqrtps xmm3, xmm3           ; xmm3 = sqrt(det)
    movaps xmm0, xmm2           ; xmm0 = b
    subps xmm0, xmm3            ; xmm0 = t = b - sqrt(det)
    ucomiss xmm0, xmm1          ; compare t against epsilon
    jb .other_sol               ; jump if t < 0
    ret
.other_sol:
    xorps xmm0, xmm0            ; xmm0 = 0
    addps xmm3, xmm2            ; t = b + sqrt(det)
    ucomiss xmm3, xmm1          ; compare t against epsilon
    jb .no_intersect            ; jump if t < 0
    movaps xmm0, xmm3           ; xmm0 = t
    ret
.no_intersect:
    xor eax, eax                ; xmm0 contains 0 already, also set ZF=1
    ret

check_intersection:
    ; Arguments:
    ; - xmm4: Ray direction
    ; - xmm5: Ray origin
    ; Returns:
    ; - Sphere address offset in edi
    ; - Intersection distance on xmm7 {t, t, t, t}
    ; Modifies: xmm0, xmm1, xmm2, xmm3, xmm7, edx, ecx, eax, edi
    mov edx, 0                  ; edx = 0
    movd xmm7, [infinity]       ; xmm7 = +inf
    shufps xmm7, xmm7, 0        ; broadcast
intersection_loop_begin:
    cmp edx, [sphere_count]     ; edx < num_spheres
    jge intersection_loop_end

    ; Each sphere consists of 14 single-precision 32-bit floating-point values
    imul ecx, edx, 14
    ; Get the 12th single-precision floating-point value (radius)
    movss xmm0, [spheres + (ecx + 12) * 4] ; xmm0 = {sphere radius, 0, 0, 0}
    shufps xmm0, xmm0, 0        ; Broadcast to all elements
    ; Get the sphere position
    movups xmm1, [spheres + ecx * 4] ; xmm1 = sphere position

    call intersect_sphere
    jz next_sphere
    ; We have an intersection, update the min t value if needed
    ucomiss xmm7, xmm0
    jb next_sphere              ; t_min < t
    movaps xmm7, xmm0           ; t_min = t
    mov edi, ecx                ; edi = sphere id
next_sphere:
    inc edx
    jmp intersection_loop_begin
intersection_loop_end:
    ret

radiance:
    ; xmm4: Ray direction
    ; xmm5: Ray origin
    call check_intersection
    ucomiss xmm7, [infinity]
    je radiance_nothing

    movups xmm0, [spheres + (edi + 8) * 4] ; xmm0 = sphere emission
    mulps xmm0, xmm9                       ; xmm0 = throughput * emission
    addps xmm8, xmm0                       ; xmm8 = radiance += throughput * emission

    movups xmm0, [spheres + (edi + 4) * 4] ; xmm0 = sphere color
    mulps xmm9, xmm0                       ; xmm9 = throughput *= color

    ; Calculate new ray origin (intersection point)
    movaps xmm0, xmm4           ; xmm0 = rd
    mulps xmm0, xmm7            ; xmm0 = rd * t
    addps xmm5, xmm0            ; xmm5 = ro + rd * t

    movups xmm1, [spheres + edi * 4] ; xmm1 = sphere center pos
    movaps xmm0, xmm5           ; xmm0 = intersection
    subps xmm0, xmm1            ; xmm0 = normal (intersection - so)
    call vec_normalize

    ; Push the ray origin a bit in the direction of the normal to prevent
    ; self-intersection artifacts.
    movss xmm1, [float_epsilon]
    shufps xmm1, xmm1, 0
    mulps xmm1, xmm0
    addps xmm5, xmm1

    ; Calculate new ray direction
    mov eax, [spheres + (edi + 13) * 4] ; eax = material id
    cmp eax, 1                          ; 1=specular, anything else=diffuse
    je specular
    ; Diffuse
    call coordinate_system
    call rand_dir_cos_weighted_hemisphere
    jmp radiance_common
specular:
    ; Perfect specular reflection
    call vec_reflect
radiance_common:
    ; Russian Roulette: randomly terminate the path with a probability inversely
    ; proportional to the throughput.
    call rand_float             ; xmm0 = rand
    movaps xmm1, xmm9           ; xmm1 = throughput
    shufps xmm1, xmm1, 0x55     ; xmm1 = throughput[1] (green)
    ucomiss xmm1, xmm0          ; compare throughput[1] with rand
    jb radiance_nothing         ; break recursion if max < rand
    ; Add missing energy to non-terminated paths
    rcpps xmm1, xmm1            ; 1 / throughput[1]
    mulps xmm9, xmm1            ; throughput *= 1 / throughput[1]

    call radiance               ; recursive
radiance_nothing:
    ret

get_raydir_for_pixel:
    ; Get the ray direction for a given pixel
    ; xmm0: x position of the pixel {x, x, x, x}
    ; xmm1: y position of the pixel {y, y, y, y}
    ; Return the ray direction in xmm0
    ; Modifies: xmm0, xmm1
    ; x * u - y * v + w_p
    mulps xmm0, xmm12
    mulps xmm1, xmm13
    subps xmm0, xmm1
    addps xmm0, xmm15
    call vec_normalize
    ret

render_pixel:
    ; Leverage the fact that we use path tracing, and consequently several
    ; samples per pixel, to get "free" antialiasing using sub-pixel sampling.
    ; Add a random [0,1) offset the pixel position.
    call rand_float             ; xmm0 = rand1
    cvtsi2ss xmm2, r9d          ; xmm2 = float(col)
    addss xmm2, xmm0            ; xmm2 = col + rand1
    shufps xmm2, xmm2, 0        ; broadcast
    call rand_float             ; xmm0 = rand2
    cvtsi2ss xmm3, r8d          ; xmm3 = float(row)
    addss xmm3, xmm0            ; xmm3 = row + rand2
    shufps xmm3, xmm3, 0        ; broadcast

    movaps xmm0, xmm2           ; xmm0 = screen pos x
    movaps xmm1, xmm3           ; xmm1 = screen pos y
    movups xmm9, [vec_ones]    ; xmm9 = throughput = {1, 1, 1, 0}

    call get_raydir_for_pixel   ; xmm0 = ray dir
    movaps xmm4, xmm0           ; xmm2 = ray dir
    movups xmm5, [camera_pos]   ; xmm3 = ray origin
    call radiance
    ret

gamma_correct:
    sqrtps xmm8, xmm8
    ret

put_pixel:
    ; Write the pixel color to memory
    ; xmm8: Pixel color
    ; r8d: Pixel row
    ; r9d: Pixel column
    ; Modified: eax, ecx, xmm9
    call gamma_correct

    movups xmm0, [vec_255]
    mulps xmm8, xmm0

    ; index = 3 * (row * width + col)
    mov ecx, r8d
    imul ecx, IMAGE_WIDTH
    add ecx, r9d
    imul ecx, 3

    cvttps2dq xmm8, xmm8      ; Convert to 32-bit ints
    packusdw xmm8, xmm8       ; Pack down to 16-bit
    packuswb xmm8, xmm8       ; Pack down to 8-bit
    movd eax, xmm8
    mov [image_array + ecx + 0], al ; 1st byte (red)
    mov [image_array + ecx + 1], ah ; 2nd byte (green)
    shr eax, 16
    mov [image_array + ecx + 2], al ; 3rd byte (blue)
    ret

render_image:
    ; Render the entire image
    ; The function is implemented as a nested loop of rows, columns and samples
    mov r8, 0                   ; row = 0
    mov eax, [image_samples]    ; xmm10 = num_samples
    cvtsi2ss xmm10, eax         ; convert to floating point
    shufps xmm10, xmm10, 0      ; broadcast to all elements
    rcpps xmm10, xmm10          ; xmm10 = 1 / num_samples
render_row:
    cmp r8, IMAGE_HEIGHT
    jge end_render_row          ; end loop if row >= image_height
    mov r9, 0                   ; col = 0
render_col:
    cmp r9, IMAGE_WIDTH
    jge end_render_col          ; end loop if col >= image_width
    mov r10, 0                  ; sample = 0
    xorps xmm8, xmm8            ; Zero-out the accumulated pixel color
render_sample:
    cmp r10, NUM_SAMPLES
    jge end_render_sample       ; end loop if sample >= num_samples
    call render_pixel           ; Render a pixel!
    inc r10
    jmp render_sample
end_render_sample:
    mulps xmm8, xmm10           ; / num_samples
    call put_pixel              ; write the pixel to memory
    inc r9
    jmp render_col
end_render_col:
    inc r8
    jmp render_row
end_render_row:
    ret

w_p:
    ; xmm12: u vector (1st column of camera-to-world matrix, or camera right vector)
    ; xmm13: v vector (2nd column of camera-to-world matrix, or camera up vector)
    ; xmm14: w vector (3rd column of camera-to-world matrix, or camera forward vector)
    ; Returns the w_p vector in xmm15
    ; Modifies: xmm0, xmm1, xmm2, xmm3, xmm4

    ; Load constants
    movd xmm0, [image_height]
    shufps xmm0, xmm0, 0        ; broadcast to all elements in the register
    cvtdq2ps xmm0, xmm0         ; convert to floating point
    movd xmm1, [image_width]
    shufps xmm1, xmm1, 0
    cvtdq2ps xmm1, xmm1
    movups xmm2, [float_05]
    shufps xmm2, xmm2, 0
    movups xmm3, [focal_length]
    shufps xmm3, xmm3, 0

    ; w_p = (-width / 2) * u + (height / 2) * v - ((height / 2) / focal_length) * w
    mulps xmm0, xmm2            ; height * 0.5
    movaps xmm4, xmm0
    mulps xmm0, xmm13           ; * v
    mulps xmm1, xmm2            ; width * 0.5
    mulps xmm1, xmm12           ; * u
    divps xmm4, xmm3            ; / focal_length
    mulps xmm4, xmm14           ; * w

    subps xmm0, xmm1
    subps xmm0, xmm4
    movaps xmm15, xmm0
    ret

_start:
    ; Seed the PRNG
    mov r14, PCG32_INIT_STATE
    mov r15, PCG32_INIT_SEQ

    ; Setup the camera vectors
    movups xmm12, [camera_right]
    movups xmm13, [camera_up]
    movups xmm14, [camera_forward]
    call w_p

    call render_image
    call print_image

    mov rax, 60           ; exit syscall
    xor rdi, rdi          ; exit code 0
    syscall

section   .data
; Bit manipulation utilities
infinity:     dd 0x7f800000
align 16                        ; Used directly in xorps
sign_bit:     dd 0x80000000
all_but_sign: dd 0x7fffffff

; Math constants
m_pi:     dd 0x40490fdb
m_2_pi:   dd 0x40c90fdb
m_inv_pi: dd 0x3ea2f983

; Constants
float_epsilon: dd 0.0001
float_0: dd 0.0
float_01: dd 0.1
float_02: dd 0.2
float_05: dd 0.5
float_08: dd 0.8
float_1: dd 1.0
float_2: dd 2.0
float_neg1: dd -1.0
vec_255: dd 255.999, 255.999, 255.999, 0.0
vec_ones: dd 1.0, 1.0, 1.0, 0.0
vec_up: dd 0.0, 1.0, 0.0, 0.0
vec_up2: dd 1.0, 0.0, 0.0, 0.0

; Image and camera related constants
image_width: dd IMAGE_WIDTH
image_height: dd IMAGE_HEIGHT
image_samples: dd NUM_SAMPLES
focal_length: dd 0.5
camera_pos: dd 0.0, 0.0, 0.0, 1.0
camera_right: dd 1.0, 0.0, 0.0, 0.0
camera_up: dd 0.0, 1.0, 0.0, 0.0
camera_forward: dd 0.0, 0.0, 1.0, 0.0

; Scene description
spheres:
    ;  Position                         Color                   Emission             Radius   Mat id
    dd     0.0, -1004.0,    -5.0, 1.0,  0.75, 0.75, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,  1000.0,  0 ; Bottom
    dd     0.0,  1004.0,    -5.0, 1.0,  0.75, 0.75, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,  1000.0,  0 ; Top
    dd  1005.0,     0.0,    -5.0, 1.0,  0.25, 0.25, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,  1000.0,  0 ; Right
    dd -1005.0,     0.0,    -5.0, 1.0,  0.75, 0.25, 0.25, 1.0,  0.0, 0.0, 0.0, 1.0,  1000.0,  0 ; Left
    dd     0.0,     0.0, -1015.0, 1.0,  0.75, 0.75, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,  1000.0,  0 ; Back
    dd   -2.25,    -2.2,   -13.0, 1.0,  0.75, 0.75, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,     1.8,  1 ; Small 1
    dd     2.0,    -2.2,   -11.0, 1.0,  0.75, 0.75, 0.75, 1.0,  0.0, 0.0, 0.0, 1.0,     1.8,  1 ; Small 2
    dd     0.0,   104.0,   -10.0, 1.0,  0.0,  0.0,  0.0,  1.0,  15.0, 15.0, 15.0, 1.0,  100.0,  0 ; Light
sphere_count: dd 8

; PBM header
header: db "P6 640 480 255", 0xa
header_length: equ $-header

; sin lookup table
%include "sin.asm"

; Memory block for the final image
section .bss
image_array: resb IMAGE_ARRAY_SIZE
