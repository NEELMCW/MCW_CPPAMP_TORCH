package = "cutorch"
version = "scm-1"

source = {
   url = "git://github.com/torch/clamptorch.git",
}

description = {
   summary = "Torch CUDA Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/clamptorch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/home/neelakandan/Downloads/mcw_cppamp/build/compiler/bin/clang -DCMAKE_CXX_COMPILER=/home/neelakandan/Downloads/mcw_cppamp/build/compiler/bin/clang++ -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}