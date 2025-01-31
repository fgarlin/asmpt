    ;; Sine function look-up table
    ;; 0 to pi/2 radians, 1025 points, single precision floating point
sine:
    dd 0x00000000, 0x3ac90fd5, 0x3b490fc6, 0x3b96cbc1, 0x3bc90f88, 0x3bfb5330, 0x3c16cb58, 0x3c2fed02
    dd 0x3c490e90, 0x3c622fff, 0x3c7b514b, 0x3c8a3938, 0x3c96c9b6, 0x3ca35a1c, 0x3cafea69, 0x3cbc7a9b
    dd 0x3cc90ab0, 0x3cd59aa6, 0x3ce22a7a, 0x3ceeba2c, 0x3cfb49ba, 0x3d03ec90, 0x3d0a342f, 0x3d107bb8
    dd 0x3d16c32c, 0x3d1d0a88, 0x3d2351cb, 0x3d2998f6, 0x3d2fe007, 0x3d3626fc, 0x3d3c6dd5, 0x3d42b491
    dd 0x3d48fb30, 0x3d4f41af, 0x3d55880e, 0x3d5bce4c, 0x3d621469, 0x3d685a62, 0x3d6ea038, 0x3d74e5e9
    dd 0x3d7b2b74, 0x3d80b86c, 0x3d83db0a, 0x3d86fd94, 0x3d8a200a, 0x3d8d426a, 0x3d9064b4, 0x3d9386e7
    dd 0x3d96a905, 0x3d99cb0a, 0x3d9cecf9, 0x3da00ecf, 0x3da3308c, 0x3da65230, 0x3da973ba, 0x3dac952b
    dd 0x3dafb680, 0x3db2d7bb, 0x3db5f8da, 0x3db919dd, 0x3dbc3ac3, 0x3dbf5b8d, 0x3dc27c39, 0x3dc59cc6
    dd 0x3dc8bd36, 0x3dcbdd86, 0x3dcefdb7, 0x3dd21dc8, 0x3dd53db9, 0x3dd85d89, 0x3ddb7d37, 0x3dde9cc4
    dd 0x3de1bc2e, 0x3de4db76, 0x3de7fa9a, 0x3deb199a, 0x3dee3876, 0x3df1572e, 0x3df475c0, 0x3df7942c
    dd 0x3dfab273, 0x3dfdd092, 0x3e007745, 0x3e02062e, 0x3e039502, 0x3e0523c2, 0x3e06b26e, 0x3e084105
    dd 0x3e09cf86, 0x3e0b5df3, 0x3e0cec4a, 0x3e0e7a8b, 0x3e1008b7, 0x3e1196cc, 0x3e1324ca, 0x3e14b2b2
    dd 0x3e164083, 0x3e17ce3d, 0x3e195be0, 0x3e1ae96b, 0x3e1c76de, 0x3e1e0438, 0x3e1f917b, 0x3e211ea5
    dd 0x3e22abb6, 0x3e2438ad, 0x3e25c58c, 0x3e275251, 0x3e28defc, 0x3e2a6b8d, 0x3e2bf804, 0x3e2d8461
    dd 0x3e2f10a2, 0x3e309cc9, 0x3e3228d4, 0x3e33b4c4, 0x3e354098, 0x3e36cc50, 0x3e3857ec, 0x3e39e36c
    dd 0x3e3b6ecf, 0x3e3cfa15, 0x3e3e853e, 0x3e401049, 0x3e419b37, 0x3e432607, 0x3e44b0b9, 0x3e463b4d
    dd 0x3e47c5c2, 0x3e495018, 0x3e4ada4f, 0x3e4c6467, 0x3e4dee60, 0x3e4f7838, 0x3e5101f1, 0x3e528b89
    dd 0x3e541501, 0x3e559e58, 0x3e57278f, 0x3e58b0a4, 0x3e5a3997, 0x3e5bc26a, 0x3e5d4b1a, 0x3e5ed3a8
    dd 0x3e605c13, 0x3e61e45c, 0x3e636c83, 0x3e64f486, 0x3e667c66, 0x3e680422, 0x3e698bba, 0x3e6b132f
    dd 0x3e6c9a7f, 0x3e6e21ab, 0x3e6fa8b2, 0x3e712f94, 0x3e72b651, 0x3e743ce8, 0x3e75c35a, 0x3e7749a6
    dd 0x3e78cfcc, 0x3e7a55cb, 0x3e7bdba4, 0x3e7d6156, 0x3e7ee6e1, 0x3e803622, 0x3e80f8c0, 0x3e81bb4a
    dd 0x3e827dc0, 0x3e834022, 0x3e840270, 0x3e84c4aa, 0x3e8586ce, 0x3e8648df, 0x3e870ada, 0x3e87ccc1
    dd 0x3e888e93, 0x3e895050, 0x3e8a11f7, 0x3e8ad38a, 0x3e8b9507, 0x3e8c566e, 0x3e8d17c0, 0x3e8dd8fc
    dd 0x3e8e9a22, 0x3e8f5b32, 0x3e901c2c, 0x3e90dd10, 0x3e919ddd, 0x3e925e94, 0x3e931f35, 0x3e93dfbf
    dd 0x3e94a031, 0x3e95608d, 0x3e9620d2, 0x3e96e100, 0x3e97a117, 0x3e986116, 0x3e9920fe, 0x3e99e0ce
    dd 0x3e9aa086, 0x3e9b6027, 0x3e9c1faf, 0x3e9cdf20, 0x3e9d9e78, 0x3e9e5db8, 0x3e9f1cdf, 0x3e9fdbee
    dd 0x3ea09ae5, 0x3ea159c2, 0x3ea21887, 0x3ea2d733, 0x3ea395c5, 0x3ea4543f, 0x3ea5129f, 0x3ea5d0e5
    dd 0x3ea68f12, 0x3ea74d25, 0x3ea80b1f, 0x3ea8c8fe, 0x3ea986c4, 0x3eaa446f, 0x3eab0201, 0x3eabbf77
    dd 0x3eac7cd4, 0x3ead3a15, 0x3eadf73c, 0x3eaeb449, 0x3eaf713a, 0x3eb02e10, 0x3eb0eacb, 0x3eb1a76b
    dd 0x3eb263ef, 0x3eb32058, 0x3eb3dca5, 0x3eb498d6, 0x3eb554ec, 0x3eb610e6, 0x3eb6ccc3, 0x3eb78884
    dd 0x3eb8442a, 0x3eb8ffb2, 0x3eb9bb1e, 0x3eba766e, 0x3ebb31a0, 0x3ebbecb6, 0x3ebca7af, 0x3ebd628b
    dd 0x3ebe1d4a, 0x3ebed7eb, 0x3ebf926f, 0x3ec04cd5, 0x3ec1071e, 0x3ec1c148, 0x3ec27b55, 0x3ec33544
    dd 0x3ec3ef15, 0x3ec4a8c8, 0x3ec5625c, 0x3ec61bd2, 0x3ec6d529, 0x3ec78e62, 0x3ec8477c, 0x3ec90077
    dd 0x3ec9b953, 0x3eca7210, 0x3ecb2aae, 0x3ecbe32c, 0x3ecc9b8b, 0x3ecd53ca, 0x3ece0bea, 0x3ecec3ea
    dd 0x3ecf7bca, 0x3ed0338a, 0x3ed0eb2a, 0x3ed1a2aa, 0x3ed25a09, 0x3ed31148, 0x3ed3c867, 0x3ed47f64
    dd 0x3ed53641, 0x3ed5ecfd, 0x3ed6a399, 0x3ed75a13, 0x3ed8106b, 0x3ed8c6a3, 0x3ed97cb9, 0x3eda32ad
    dd 0x3edae880, 0x3edb9e31, 0x3edc53c1, 0x3edd092e, 0x3eddbe79, 0x3ede73a2, 0x3edf28a9, 0x3edfdd8d
    dd 0x3ee0924f, 0x3ee146ee, 0x3ee1fb6a, 0x3ee2afc4, 0x3ee363fa, 0x3ee4180e, 0x3ee4cbfe, 0x3ee57fcb
    dd 0x3ee63375, 0x3ee6e6fb, 0x3ee79a5d, 0x3ee84d9c, 0x3ee900b7, 0x3ee9b3ae, 0x3eea6681, 0x3eeb1930
    dd 0x3eebcbbb, 0x3eec7e21, 0x3eed3063, 0x3eede280, 0x3eee9479, 0x3eef464c, 0x3eeff7fb, 0x3ef0a985
    dd 0x3ef15aea, 0x3ef20c29, 0x3ef2bd43, 0x3ef36e38, 0x3ef41f07, 0x3ef4cfb1, 0x3ef58035, 0x3ef63093
    dd 0x3ef6e0cb, 0x3ef790dc, 0x3ef840c8, 0x3ef8f08e, 0x3ef9a02d, 0x3efa4fa5, 0x3efafef7, 0x3efbae22
    dd 0x3efc5d27, 0x3efd0c04, 0x3efdbabb, 0x3efe694a, 0x3eff17b2, 0x3effc5f3, 0x3f003a06, 0x3f0090ff
    dd 0x3f00e7e4, 0x3f013eb5, 0x3f019573, 0x3f01ec1c, 0x3f0242b1, 0x3f029932, 0x3f02ef9f, 0x3f0345f8
    dd 0x3f039c3d, 0x3f03f26d, 0x3f044889, 0x3f049e91, 0x3f04f484, 0x3f054a62, 0x3f05a02c, 0x3f05f5e2
    dd 0x3f064b82, 0x3f06a10e, 0x3f06f686, 0x3f074be8, 0x3f07a136, 0x3f07f66f, 0x3f084b92, 0x3f08a0a1
    dd 0x3f08f59b, 0x3f094a7f, 0x3f099f4e, 0x3f09f409, 0x3f0a48ad, 0x3f0a9d3d, 0x3f0af1b7, 0x3f0b461c
    dd 0x3f0b9a6b, 0x3f0beea5, 0x3f0c42c9, 0x3f0c96d7, 0x3f0cead0, 0x3f0d3eb3, 0x3f0d9281, 0x3f0de638
    dd 0x3f0e39da, 0x3f0e8d65, 0x3f0ee0db, 0x3f0f343b, 0x3f0f8784, 0x3f0fdab8, 0x3f102dd5, 0x3f1080dc
    dd 0x3f10d3cd, 0x3f1126a7, 0x3f11796b, 0x3f11cc19, 0x3f121eb0, 0x3f127130, 0x3f12c39a, 0x3f1315ee
    dd 0x3f13682a, 0x3f13ba50, 0x3f140c5f, 0x3f145e58, 0x3f14b039, 0x3f150204, 0x3f1553b7, 0x3f15a554
    dd 0x3f15f6d9, 0x3f164847, 0x3f16999f, 0x3f16eade, 0x3f173c07, 0x3f178d18, 0x3f17de12, 0x3f182ef5
    dd 0x3f187fc0, 0x3f18d073, 0x3f19210f, 0x3f197194, 0x3f19c200, 0x3f1a1255, 0x3f1a6293, 0x3f1ab2b8
    dd 0x3f1b02c6, 0x3f1b52bb, 0x3f1ba299, 0x3f1bf25f, 0x3f1c420c, 0x3f1c91a2, 0x3f1ce11f, 0x3f1d3084
    dd 0x3f1d7fd1, 0x3f1dcf06, 0x3f1e1e22, 0x3f1e6d26, 0x3f1ebc12, 0x3f1f0ae5, 0x3f1f599f, 0x3f1fa841
    dd 0x3f1ff6cb, 0x3f20453b, 0x3f209393, 0x3f20e1d2, 0x3f212ff9, 0x3f217e06, 0x3f21cbfb, 0x3f2219d7
    dd 0x3f226799, 0x3f22b543, 0x3f2302d3, 0x3f23504b, 0x3f239da9, 0x3f23eaee, 0x3f24381a, 0x3f24852c
    dd 0x3f24d225, 0x3f251f04, 0x3f256bcb, 0x3f25b877, 0x3f26050a, 0x3f265184, 0x3f269de3, 0x3f26ea2a
    dd 0x3f273656, 0x3f278268, 0x3f27ce61, 0x3f281a40, 0x3f286605, 0x3f28b1b0, 0x3f28fd41, 0x3f2948b8
    dd 0x3f299415, 0x3f29df57, 0x3f2a2a80, 0x3f2a758e, 0x3f2ac082, 0x3f2b0b5b, 0x3f2b561b, 0x3f2ba0bf
    dd 0x3f2beb4a, 0x3f2c35b9, 0x3f2c800f, 0x3f2cca49, 0x3f2d1469, 0x3f2d5e6f, 0x3f2da859, 0x3f2df229
    dd 0x3f2e3bde, 0x3f2e8578, 0x3f2ecef7, 0x3f2f185b, 0x3f2f61a5, 0x3f2faad3, 0x3f2ff3e6, 0x3f303cde
    dd 0x3f3085bb, 0x3f30ce7c, 0x3f311722, 0x3f315fad, 0x3f31a81d, 0x3f31f071, 0x3f3238aa, 0x3f3280c7
    dd 0x3f32c8c9, 0x3f3310af, 0x3f33587a, 0x3f33a029, 0x3f33e7bc, 0x3f342f34, 0x3f34768f, 0x3f34bdcf
    dd 0x3f3504f3, 0x3f354bfb, 0x3f3592e7, 0x3f35d9b8, 0x3f36206c, 0x3f366704, 0x3f36ad7f, 0x3f36f3df
    dd 0x3f373a23, 0x3f37804a, 0x3f37c655, 0x3f380c43, 0x3f385216, 0x3f3897cb, 0x3f38dd65, 0x3f3922e1
    dd 0x3f396842, 0x3f39ad85, 0x3f39f2ac, 0x3f3a37b7, 0x3f3a7ca4, 0x3f3ac175, 0x3f3b0629, 0x3f3b4ac1
    dd 0x3f3b8f3b, 0x3f3bd398, 0x3f3c17d9, 0x3f3c5bfc, 0x3f3ca003, 0x3f3ce3ec, 0x3f3d27b8, 0x3f3d6b67
    dd 0x3f3daef9, 0x3f3df26e, 0x3f3e35c5, 0x3f3e78ff, 0x3f3ebc1b, 0x3f3eff1b, 0x3f3f41fc, 0x3f3f84c0
    dd 0x3f3fc767, 0x3f4009f0, 0x3f404c5c, 0x3f408ea9, 0x3f40d0da, 0x3f4112ec, 0x3f4154e1, 0x3f4196b7
    dd 0x3f41d870, 0x3f421a0b, 0x3f425b89, 0x3f429ce8, 0x3f42de29, 0x3f431f4c, 0x3f436051, 0x3f43a138
    dd 0x3f43e200, 0x3f4422ab, 0x3f446337, 0x3f44a3a5, 0x3f44e3f5, 0x3f452426, 0x3f456439, 0x3f45a42d
    dd 0x3f45e403, 0x3f4623bb, 0x3f466354, 0x3f46a2ce, 0x3f46e22a, 0x3f472167, 0x3f476085, 0x3f479f84
    dd 0x3f47de65, 0x3f481d27, 0x3f485bca, 0x3f489a4e, 0x3f48d8b3, 0x3f4916fa, 0x3f495521, 0x3f499329
    dd 0x3f49d112, 0x3f4a0edc, 0x3f4a4c87, 0x3f4a8a13, 0x3f4ac77f, 0x3f4b04cc, 0x3f4b41fa, 0x3f4b7f09
    dd 0x3f4bbbf8, 0x3f4bf8c7, 0x3f4c3578, 0x3f4c7208, 0x3f4cae79, 0x3f4ceacb, 0x3f4d26fd, 0x3f4d6310
    dd 0x3f4d9f02, 0x3f4ddad5, 0x3f4e1689, 0x3f4e521c, 0x3f4e8d90, 0x3f4ec8e4, 0x3f4f0417, 0x3f4f3f2b
    dd 0x3f4f7a1f, 0x3f4fb4f4, 0x3f4fefa8, 0x3f502a3b, 0x3f5064af, 0x3f509f03, 0x3f50d937, 0x3f51134a
    dd 0x3f514d3d, 0x3f518710, 0x3f51c0c2, 0x3f51fa54, 0x3f5233c6, 0x3f526d18, 0x3f52a649, 0x3f52df59
    dd 0x3f531849, 0x3f535118, 0x3f5389c7, 0x3f53c255, 0x3f53fac3, 0x3f54330f, 0x3f546b3b, 0x3f54a347
    dd 0x3f54db31, 0x3f5512fb, 0x3f554aa4, 0x3f55822c, 0x3f55b993, 0x3f55f0d9, 0x3f5627fe, 0x3f565f02
    dd 0x3f5695e5, 0x3f56cca7, 0x3f570348, 0x3f5739c7, 0x3f577026, 0x3f57a663, 0x3f57dc7f, 0x3f581279
    dd 0x3f584853, 0x3f587e0b, 0x3f58b3a1, 0x3f58e916, 0x3f591e6a, 0x3f59539c, 0x3f5988ad, 0x3f59bd9c
    dd 0x3f59f26a, 0x3f5a2716, 0x3f5a5ba0, 0x3f5a9009, 0x3f5ac450, 0x3f5af875, 0x3f5b2c79, 0x3f5b605a
    dd 0x3f5b941a, 0x3f5bc7b8, 0x3f5bfb34, 0x3f5c2e8e, 0x3f5c61c7, 0x3f5c94dd, 0x3f5cc7d1, 0x3f5cfaa3
    dd 0x3f5d2d53, 0x3f5d5fe1, 0x3f5d924d, 0x3f5dc497, 0x3f5df6be, 0x3f5e28c3, 0x3f5e5aa6, 0x3f5e8c67
    dd 0x3f5ebe05, 0x3f5eef81, 0x3f5f20db, 0x3f5f5212, 0x3f5f8327, 0x3f5fb419, 0x3f5fe4e9, 0x3f601596
    dd 0x3f604621, 0x3f607689, 0x3f60a6cf, 0x3f60d6f2, 0x3f6106f2, 0x3f6136d0, 0x3f61668a, 0x3f619622
    dd 0x3f61c598, 0x3f61f4ea, 0x3f62241a, 0x3f625326, 0x3f628210, 0x3f62b0d7, 0x3f62df7b, 0x3f630dfc
    dd 0x3f633c5a, 0x3f636a95, 0x3f6398ac, 0x3f63c6a1, 0x3f63f473, 0x3f642221, 0x3f644fac, 0x3f647d14
    dd 0x3f64aa59, 0x3f64d77b, 0x3f650479, 0x3f653154, 0x3f655e0b, 0x3f658aa0, 0x3f65b710, 0x3f65e35e
    dd 0x3f660f88, 0x3f663b8e, 0x3f666771, 0x3f669330, 0x3f66becc, 0x3f66ea45, 0x3f671599, 0x3f6740ca
    dd 0x3f676bd8, 0x3f6796c1, 0x3f67c187, 0x3f67ec29, 0x3f6816a8, 0x3f684103, 0x3f686b39, 0x3f68954c
    dd 0x3f68bf3c, 0x3f68e907, 0x3f6912ae, 0x3f693c32, 0x3f696591, 0x3f698ecc, 0x3f69b7e4, 0x3f69e0d7
    dd 0x3f6a09a7, 0x3f6a3252, 0x3f6a5ad9, 0x3f6a833c, 0x3f6aab7b, 0x3f6ad395, 0x3f6afb8c, 0x3f6b235e
    dd 0x3f6b4b0c, 0x3f6b7295, 0x3f6b99fb, 0x3f6bc13b, 0x3f6be858, 0x3f6c0f50, 0x3f6c3624, 0x3f6c5cd4
    dd 0x3f6c835e, 0x3f6ca9c5, 0x3f6cd007, 0x3f6cf624, 0x3f6d1c1d, 0x3f6d41f2, 0x3f6d67a1, 0x3f6d8d2d
    dd 0x3f6db293, 0x3f6dd7d5, 0x3f6dfcf2, 0x3f6e21eb, 0x3f6e46be, 0x3f6e6b6d, 0x3f6e8ff8, 0x3f6eb45d
    dd 0x3f6ed89e, 0x3f6efcba, 0x3f6f20b0, 0x3f6f4483, 0x3f6f6830, 0x3f6f8bb8, 0x3f6faf1b, 0x3f6fd25a
    dd 0x3f6ff573, 0x3f701867, 0x3f703b37, 0x3f705de1, 0x3f708066, 0x3f70a2c6, 0x3f70c501, 0x3f70e717
    dd 0x3f710908, 0x3f712ad4, 0x3f714c7a, 0x3f716dfb, 0x3f718f57, 0x3f71b08e, 0x3f71d19f, 0x3f71f28c
    dd 0x3f721352, 0x3f7233f4, 0x3f725470, 0x3f7274c7, 0x3f7294f8, 0x3f72b504, 0x3f72d4eb, 0x3f72f4ac
    dd 0x3f731447, 0x3f7333be, 0x3f73530e, 0x3f737239, 0x3f73913f, 0x3f73b01f, 0x3f73ced9, 0x3f73ed6e
    dd 0x3f740bdd, 0x3f742a27, 0x3f74484b, 0x3f746649, 0x3f748422, 0x3f74a1d5, 0x3f74bf62, 0x3f74dcc9
    dd 0x3f74fa0b, 0x3f751727, 0x3f75341d, 0x3f7550ed, 0x3f756d97, 0x3f758a1c, 0x3f75a67b, 0x3f75c2b3
    dd 0x3f75dec6, 0x3f75fab3, 0x3f76167a, 0x3f76321b, 0x3f764d97, 0x3f7668ec, 0x3f76841b, 0x3f769f24
    dd 0x3f76ba07, 0x3f76d4c4, 0x3f76ef5b, 0x3f7709cc, 0x3f772417, 0x3f773e3c, 0x3f77583a, 0x3f777213
    dd 0x3f778bc5, 0x3f77a551, 0x3f77beb7, 0x3f77d7f7, 0x3f77f110, 0x3f780a04, 0x3f7822d1, 0x3f783b77
    dd 0x3f7853f8, 0x3f786c52, 0x3f788486, 0x3f789c93, 0x3f78b47b, 0x3f78cc3b, 0x3f78e3d6, 0x3f78fb4a
    dd 0x3f791298, 0x3f7929bf, 0x3f7940c0, 0x3f79579a, 0x3f796e4e, 0x3f7984dc, 0x3f799b43, 0x3f79b183
    dd 0x3f79c79d, 0x3f79dd91, 0x3f79f35e, 0x3f7a0904, 0x3f7a1e84, 0x3f7a33dd, 0x3f7a4910, 0x3f7a5e1c
    dd 0x3f7a7302, 0x3f7a87c1, 0x3f7a9c59, 0x3f7ab0cb, 0x3f7ac516, 0x3f7ad93a, 0x3f7aed37, 0x3f7b010e
    dd 0x3f7b14be, 0x3f7b2848, 0x3f7b3bab, 0x3f7b4ee7, 0x3f7b61fc, 0x3f7b74ea, 0x3f7b87b2, 0x3f7b9a53
    dd 0x3f7baccd, 0x3f7bbf20, 0x3f7bd14d, 0x3f7be353, 0x3f7bf531, 0x3f7c06e9, 0x3f7c187a, 0x3f7c29e5
    dd 0x3f7c3b28, 0x3f7c4c44, 0x3f7c5d3a, 0x3f7c6e08, 0x3f7c7eb0, 0x3f7c8f31, 0x3f7c9f8a, 0x3f7cafbd
    dd 0x3f7cbfc9, 0x3f7ccfae, 0x3f7cdf6c, 0x3f7cef03, 0x3f7cfe73, 0x3f7d0dbc, 0x3f7d1cdd, 0x3f7d2bd8
    dd 0x3f7d3aac, 0x3f7d4959, 0x3f7d57de, 0x3f7d663d, 0x3f7d7474, 0x3f7d8285, 0x3f7d906e, 0x3f7d9e30
    dd 0x3f7dabcc, 0x3f7db940, 0x3f7dc68c, 0x3f7dd3b2, 0x3f7de0b1, 0x3f7ded88, 0x3f7dfa38, 0x3f7e06c2
    dd 0x3f7e1324, 0x3f7e1f5e, 0x3f7e2b72, 0x3f7e375e, 0x3f7e4323, 0x3f7e4ec1, 0x3f7e5a38, 0x3f7e6588
    dd 0x3f7e70b0, 0x3f7e7bb1, 0x3f7e868b, 0x3f7e913d, 0x3f7e9bc9, 0x3f7ea62d, 0x3f7eb069, 0x3f7eba7f
    dd 0x3f7ec46d, 0x3f7ece34, 0x3f7ed7d4, 0x3f7ee14c, 0x3f7eea9d, 0x3f7ef3c7, 0x3f7efcc9, 0x3f7f05a4
    dd 0x3f7f0e58, 0x3f7f16e4, 0x3f7f1f49, 0x3f7f2787, 0x3f7f2f9d, 0x3f7f378c, 0x3f7f3f54, 0x3f7f46f4
    dd 0x3f7f4e6d, 0x3f7f55bf, 0x3f7f5ce9, 0x3f7f63ec, 0x3f7f6ac7, 0x3f7f717b, 0x3f7f7808, 0x3f7f7e6d
    dd 0x3f7f84ab, 0x3f7f8ac2, 0x3f7f90b1, 0x3f7f9678, 0x3f7f9c18, 0x3f7fa191, 0x3f7fa6e3, 0x3f7fac0d
    dd 0x3f7fb10f, 0x3f7fb5ea, 0x3f7fba9e, 0x3f7fbf2a, 0x3f7fc38f, 0x3f7fc7cc, 0x3f7fcbe2, 0x3f7fcfd1
    dd 0x3f7fd397, 0x3f7fd737, 0x3f7fdaaf, 0x3f7fde00, 0x3f7fe129, 0x3f7fe42b, 0x3f7fe705, 0x3f7fe9b8
    dd 0x3f7fec43, 0x3f7feea7, 0x3f7ff0e3, 0x3f7ff2f8, 0x3f7ff4e6, 0x3f7ff6ac, 0x3f7ff84a, 0x3f7ff9c1
    dd 0x3f7ffb11, 0x3f7ffc39, 0x3f7ffd39, 0x3f7ffe13, 0x3f7ffec4, 0x3f7fff4e, 0x3f7fffb1, 0x3f7fffec
    dd 0x3f800000
