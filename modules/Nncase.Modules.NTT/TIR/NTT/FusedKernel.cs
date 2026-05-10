// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Explicit target-level fused kernel marker selected before TIR codegen.
/// </summary>
public sealed partial class FusedKernel : NTTKernelOp
{
    public static readonly ParameterInfo Arg0 = new(typeof(FusedKernel), 0, "arg0");

    public static readonly ParameterInfo Arg1 = new(typeof(FusedKernel), 1, "arg1");

    public static readonly ParameterInfo Arg2 = new(typeof(FusedKernel), 2, "arg2");

    public static readonly ParameterInfo Output = new(typeof(FusedKernel), 3, "output");

    public string FusionName { get; }

    public override string DisplayProperty() => $"FusionName: {FusionName}";
}
