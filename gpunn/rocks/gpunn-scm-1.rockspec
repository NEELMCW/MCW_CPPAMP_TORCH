package = "gpunn"
version = "scm-1"

source = {
   url = "git://github.com/torch/gpunn.git",
}

description = {
   summary = "Torch GPU Neural Network Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/gpunn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "gputorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${MCWCPPAMPROOT}/gmac_exp_build/compiler/bin/clang -DCMAKE_CXX_COMPILER=${MCWCPPAMPROOT}/gmac_exp_build/compiler/bin/clang++ -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
