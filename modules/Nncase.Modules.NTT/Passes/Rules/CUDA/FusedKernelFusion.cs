// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CUDA;

/// <summary>
/// Fuse a dense attention compute chain into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaFlashAttention : FusionMaker
{
    public override string Name => "cuda.flash_attention";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var q = IsWildcard("q") with { TypePattern = IsFloat() };
        var k = IsWildcard("k") with { TypePattern = IsFloat() };
        var v = IsWildcard("v") with { TypePattern = IsFloat() };
        var kTranspose = IsTranspose(
            "kTranspose",
            "kTransposeCall",
            _ => true,
            k,
            IsWildcard("kTransposePerm"));
        var qk = IsMatMul(
            "qkMatmul",
            "qkMatmulCall",
            _ => true,
            q,
            kTranspose,
            IsWildcard());
        var scale = IsTensorConst() with { TypePattern = IsScalar() | HasShape(new Dimension[] { new DimConst(1) }) };
        var scaledQk = IsSwappableBinary(null!, "scaledQkCall", b => b.BinaryOp == BinaryOp.Mul, qk, scale);
        var probs = IsSoftmax("attentionSoftmax", scaledQk, IsWildcard("axis"));
        return IsMatMul(
            "avMatmul",
            "root",
            _ => true,
            probs,
            v,
            IsWildcard());
    }

    private Call? GetReplace(Call root, Expr q, Expr k, Expr v)
    {
        var parameters = new List<Expr>();
        var vars = new List<Var>();
        var varMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance);

        AddParameter(q, "q");
        AddParameter(k, "k");
        AddParameter(v, "v");

        var clonedRoot = new FusionMerger(varMap).Clone(root, default);
        var fusion = new Fusion($"{Name}_{Count++}", ModuleKind, clonedRoot, vars.ToArray());
        return new Call(fusion, parameters.ToArray());

        void AddParameter(Expr expr, string name)
        {
            if (varMap.ContainsKey(expr))
            {
                return;
            }

            var @var = new Var(name, expr.CheckedType!);
            varMap.Add(expr, @var);
            vars.Add(@var);
            parameters.Add(expr);
        }
    }
}
