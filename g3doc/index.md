# Orbax

<!-- BEGIN GOOGLE-INTERNAL -->

go/orbax

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'cpgaffney' reviewed: '2022-05-10' }
*-->

<!-- END GOOGLE-INTERNAL -->

[TOC]

## Introduction

Orbax is a library providing training utilities for JAX users. At the moment, it
only provides a checkpointing library, detailed below.

<!-- BEGIN GOOGLE-INTERNAL -->

See [T5X Controller](http://go/t5x-controller) for details on the original
design doc. <!-- END GOOGLE-INTERNAL -->

## Reference

### [Checkpointing](checkpoint.md)

This documentation illustrates how checkpointing works using Orbax.
