import os, subprocess

def options(opt):
    opt.load(['compiler_cxx','gnu_dirs','cxx14','boost','gmpxx','mpfr',
              'elemental','libxml2', 'rapidjson'])

def configure(conf):
    if not 'CXX' in os.environ or os.environ['CXX']=='g++' or os.environ['CXX']=='icpc':
        conf.environ['CXX']='mpicxx'

    conf.load(['compiler_cxx','gnu_dirs','cxx14','boost','gmpxx','mpfr',
               'elemental','libxml2', 'rapidjson'])

    conf.env.git_version=subprocess.check_output('git describe --dirty', universal_newlines=True, shell=True).rstrip()
    
def build(bld):
    default_flags=['-Wall', '-Wextra', '-O3', '-D SDPB_VERSION_STRING="' + bld.env.git_version + '"']
    # default_flags=['-Wall', '-Wextra', '-g', '-D SDPB_VERSION_STRING="' + bld.env.git_version + '"']
    use_packages=['cxx14','boost','gmpxx','mpfr','elemental','libxml2', 'rapidjson']
    
    bld.program(source=['src/outer/main.cxx'],
                target='outer',
                cxxflags=default_flags,
                use=use_packages
                )

    
