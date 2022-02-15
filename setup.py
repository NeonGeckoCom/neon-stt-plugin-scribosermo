#!/usr/bin/env python3
from setuptools import setup

PLUGIN_ENTRY_POINT = 'neon-stt-plugin-scribosermo=neon_stt_plugin_scribosermo:ScriboSermoSTT'
setup(
    name='neon-stt-plugin-scribosermo',
    version='0.0.1',
    description='A scribosermo stt plugin for mycroft neon ovos',
    url='https://github.com/NeonGeckoCom/neon-stt-plugin-scribosermo',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    license='Apache-2.0',
    packages=['neon_stt_plugin_scribosermo'],
    install_requires=["ds-ctcdecoder",
                      "tflite-runtime",
                      "ovos-plugin-manager>=0.0.1a7"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='mycroft ovos neon plugin stt',
    entry_points={'mycroft.plugin.stt': PLUGIN_ENTRY_POINT}
)
