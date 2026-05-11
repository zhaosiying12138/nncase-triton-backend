// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Distributed;
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
        var q = CudaFusionRuleUtility.MaybeBoxed("q", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var k = CudaFusionRuleUtility.MaybeBoxed("k", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var v = CudaFusionRuleUtility.MaybeBoxed("v", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var kTranspose = CudaFusionRuleUtility.MaybeBoxed(IsTranspose(
            "kTranspose",
            "kTransposeCall",
            _ => true,
            k,
            IsWildcard("kTransposePerm")));
        var qk = CudaFusionRuleUtility.MaybeBoxed(IsMatMul(
            "qkMatmul",
            "qkMatmulCall",
            _ => true,
            q,
            kTranspose,
            IsWildcard()));
        var scale = IsTensorConst() with { TypePattern = CudaFusionRuleUtility.IsScalarLike() | HasShape(new Dimension[] { new DimConst(1) }) };
        var scaledQk = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary(null!, "scaledQkCall", b => b.BinaryOp == BinaryOp.Mul, qk, scale));
        var probs = CudaFusionRuleUtility.MaybeBoxed(IsSoftmax("attentionSoftmax", scaledQk, IsWildcard("axis")));
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
        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (q, "q"),
            (k, "k"),
            (v, "v"));
    }
}

/// <summary>
/// Fuse swish(gate) * up into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaSwishMul : FusionMaker
{
    public override string Name => "cuda.swish_mul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var gate = CudaFusionRuleUtility.MaybeBoxed("gate", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var up = CudaFusionRuleUtility.MaybeBoxed("up", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var swish = CudaFusionRuleUtility.MaybeBoxed(IsSwish(
            "swish",
            "swishCall",
            _ => true,
            gate,
            IsTensorConst("beta") with { TypePattern = IsFloatScalar() }));
        return IsSwappableBinary("mul", "root", binary => binary.BinaryOp == BinaryOp.Mul, swish, up);
    }

    private Call? GetReplace(Call root, Expr gate, Expr up, float beta)
    {
        if (!CudaFusionRuleUtility.IsDefaultSwishBeta(beta))
        {
            return null;
        }

        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (gate, "gate"),
            (up, "up"));
    }
}

/// <summary>
/// Fuse swish(gate) * matmul(lhs, rhs) into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaMatmulSwishMul : FusionMaker
{
    public override string Name => "cuda.matmul_swish_mul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var gate = CudaFusionRuleUtility.MaybeBoxed("gate", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var lhs = CudaFusionRuleUtility.MaybeBoxed("lhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var rhs = CudaFusionRuleUtility.MaybeBoxed("rhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var swish = CudaFusionRuleUtility.MaybeBoxed(IsSwish(
            "swish",
            "swishCall",
            _ => true,
            gate,
            IsTensorConst("beta") with { TypePattern = IsFloatScalar() }));
        var matmul = CudaFusionRuleUtility.MaybeBoxed(IsMatMul("upMatmul", "upMatmulCall", _ => true, lhs, rhs, IsWildcard()));
        return IsSwappableBinary("mul", "root", binary => binary.BinaryOp == BinaryOp.Mul, swish, matmul);
    }

    private Call? GetReplace(Call root, Expr gate, Expr lhs, Expr rhs, float beta)
    {
        if (!CudaFusionRuleUtility.IsDefaultSwishBeta(beta))
        {
            return null;
        }

        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (gate, "gate"),
            (lhs, "lhs"),
            (rhs, "rhs"));
    }
}

/// <summary>
/// Fuse matmul(swish(gate) * up, rhs) into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaSiluMulMatmul : FusionMaker
{
    public override string Name => "cuda.silu_mul_matmul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var gate = CudaFusionRuleUtility.MaybeBoxed("gate", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var up = CudaFusionRuleUtility.MaybeBoxed("up", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var rhs = CudaFusionRuleUtility.MaybeBoxed("rhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var swish = CudaFusionRuleUtility.MaybeBoxed(IsSwish(
            "swish",
            "swishCall",
            _ => true,
            gate,
            IsTensorConst("beta") with { TypePattern = IsFloatScalar() }));
        var siluUp = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary("siluUp", "siluUpCall", binary => binary.BinaryOp == BinaryOp.Mul, swish, up));
        return IsMatMul("downMatmul", "root", _ => true, siluUp, rhs, IsWildcard());
    }

    private Call? GetReplace(Call root, Expr gate, Expr up, Expr rhs, float beta)
    {
        if (!CudaFusionRuleUtility.IsDefaultSwishBeta(beta))
        {
            return null;
        }

        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (gate, "gate"),
            (up, "up"),
            (rhs, "rhs"));
    }
}

/// <summary>
/// Fuse matmul(gate * matmul(lhs, up_rhs), down_rhs) into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaMatmulMulMatmul : FusionMaker
{
    public override string Name => "cuda.matmul_mul_matmul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var gate = CudaFusionRuleUtility.MaybeBoxed("gate", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var lhs = CudaFusionRuleUtility.MaybeBoxed("lhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var upRhs = CudaFusionRuleUtility.MaybeBoxed("upRhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var downRhs = CudaFusionRuleUtility.MaybeBoxed("downRhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var up = CudaFusionRuleUtility.MaybeBoxed(IsMatMul("upMatmul", "upMatmulCall", _ => true, lhs, upRhs, IsWildcard()));
        var gated = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary("gated", "gatedCall", binary => binary.BinaryOp == BinaryOp.Mul, gate, up));
        return IsMatMul("downMatmul", "root", _ => true, gated, downRhs, IsWildcard());
    }

    private Call? GetReplace(Call root, Expr gate, Expr lhs, Expr upRhs, Expr downRhs)
    {
        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (gate, "gate"),
            (lhs, "lhs"),
            (upRhs, "up_rhs"),
            (downRhs, "down_rhs"));
    }
}

/// <summary>
/// Fuse matmul(swish(matmul(lhs, gate_rhs)) * matmul(lhs, up_rhs), down_rhs) into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaMatmulSiluMatmulMulMatmul : FusionMaker
{
    public override string Name => "cuda.matmul_silu_matmul_mul_matmul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var gateLhs = CudaFusionRuleUtility.MaybeBoxed("gateLhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var upLhs = CudaFusionRuleUtility.MaybeBoxed("upLhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var gateRhs = CudaFusionRuleUtility.MaybeBoxed("gateRhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var upRhs = CudaFusionRuleUtility.MaybeBoxed("upRhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var downRhs = CudaFusionRuleUtility.MaybeBoxed("downRhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var gate = CudaFusionRuleUtility.MaybeBoxed(IsMatMul("gateMatmul", "gateMatmulCall", _ => true, gateLhs, gateRhs, IsWildcard()));
        var up = CudaFusionRuleUtility.MaybeBoxed(IsMatMul("upMatmul", "upMatmulCall", _ => true, upLhs, upRhs, IsWildcard()));
        var swish = CudaFusionRuleUtility.MaybeBoxed(IsSwish(
            "swish",
            "swishCall",
            _ => true,
            gate,
            IsTensorConst("beta") with { TypePattern = IsFloatScalar() }));
        var hidden = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary("hidden", "hiddenCall", binary => binary.BinaryOp == BinaryOp.Mul, swish, up));
        return IsMatMul("downMatmul", "root", _ => true, hidden, downRhs, IsWildcard());
    }

    private Call? GetReplace(Call root, Expr gateLhs, Expr upLhs, Expr gateRhs, Expr upRhs, Expr downRhs, float beta)
    {
        if (!CudaFusionRuleUtility.IsDefaultSwishBeta(beta))
        {
            return null;
        }

        if (!CudaFusionRuleUtility.TryGetEquivalentInput(gateLhs, upLhs, out var lhs))
        {
            return null;
        }

        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (lhs, "lhs"),
            (gateRhs, "gate_rhs"),
            (upRhs, "up_rhs"),
            (downRhs, "down_rhs"));
    }
}

/// <summary>
/// Fuse layer_norm(input) -> transpose into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaLayerNormTranspose : FusionMaker
{
    public override string Name => "cuda.layer_norm_transpose";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var input = CudaFusionRuleUtility.MaybeBoxed("input", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var scale = CudaFusionRuleUtility.MaybeBoxed("scale", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var bias = CudaFusionRuleUtility.MaybeBoxed("bias", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var layerNorm = CudaFusionRuleUtility.MaybeBoxed(IsLayerNorm("layerNorm", "layerNormCall", _ => true, input, scale, bias));
        return IsTranspose("transpose", "root", _ => true, layerNorm, IsFixedShape("perm"));
    }

    private Call? GetReplace(Call root, LayerNorm layerNorm, Expr input, Expr scale, Expr bias, int[] perm)
    {
        var fusionName = CudaFusionRuleUtility.EncodeLayerNormTransposeName(Name, Count++, layerNorm, perm);
        return CudaFusionRuleUtility.MakeFusionCall(
            fusionName,
            ModuleKind,
            root,
            (input, "input"),
            (scale, "scale"),
            (bias, "bias"));
    }
}

/// <summary>
/// Fuse layer_norm(input) -> matmul into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaLayerNormMatmul : FusionMaker
{
    public override string Name => "cuda.layer_norm_matmul";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var input = CudaFusionRuleUtility.MaybeBoxed("input", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var scale = CudaFusionRuleUtility.MaybeBoxed("scale", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var bias = CudaFusionRuleUtility.MaybeBoxed("bias", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var rhs = CudaFusionRuleUtility.MaybeBoxed("rhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var layerNorm = CudaFusionRuleUtility.MaybeBoxed(IsLayerNorm("layerNorm", "layerNormCall", _ => true, input, scale, bias));
        return IsMatMul("matmul", "root", _ => true, layerNorm, rhs, IsWildcard());
    }

    private Call? GetReplace(Call root, LayerNorm layerNorm, Expr input, Expr scale, Expr bias, Expr rhs)
    {
        var fusionName = CudaFusionRuleUtility.EncodeLayerNormName(Name, Count++, layerNorm);
        return CudaFusionRuleUtility.MakeFusionCall(
            fusionName,
            ModuleKind,
            root,
            (input, "input"),
            (scale, "scale"),
            (bias, "bias"),
            (rhs, "rhs"));
    }
}

/// <summary>
/// Fuse mul(input, scale) -> cos into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaMulCos : FusionMaker
{
    public override string Name => "cuda.mul_cos";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var lhs = CudaFusionRuleUtility.MaybeBoxed("lhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var rhs = CudaFusionRuleUtility.MaybeBoxed("rhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var mul = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary("mul", "mulCall", binary => binary.BinaryOp == BinaryOp.Mul, lhs, rhs));
        return IsUnary("cos", "root", unary => unary.UnaryOp == UnaryOp.Cos, mul);
    }

    private Call? GetReplace(Call root, Expr lhs, Expr rhs)
    {
        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (lhs, "lhs"),
            (rhs, "rhs"));
    }
}

/// <summary>
/// Fuse mul(input, scale) -> sin into an explicit CUDA Fusion marker.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaMulSin : FusionMaker
{
    public override string Name => "cuda.mul_sin";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var lhs = CudaFusionRuleUtility.MaybeBoxed("lhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var rhs = CudaFusionRuleUtility.MaybeBoxed("rhs", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var mul = CudaFusionRuleUtility.MaybeBoxed(IsSwappableBinary("mul", "mulCall", binary => binary.BinaryOp == BinaryOp.Mul, lhs, rhs));
        return IsUnary("sin", "root", unary => unary.UnaryOp == UnaryOp.Sin, mul);
    }

    private Call? GetReplace(Call root, Expr lhs, Expr rhs)
    {
        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (lhs, "lhs"),
            (rhs, "rhs"));
    }
}

/// <summary>
/// Fuse RoPE into an explicit CUDA Fusion marker for compute-mode native Triton lowering.
/// </summary>
[RuleGenerator]
public sealed partial class FuseCudaRoPE : FusionMaker
{
    public override string Name => "cuda.rope";

    public override string ModuleKind => CUDATarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var input = CudaFusionRuleUtility.MaybeBoxed("input", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var cos = CudaFusionRuleUtility.MaybeBoxed("cos", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        var sin = CudaFusionRuleUtility.MaybeBoxed("sin", IsWildcard() with { TypePattern = CudaFusionRuleUtility.IsFloatLike() });
        return IsRoPE("rope", "root", _ => true, input, cos, sin);
    }

    private Call? GetReplace(Call root, Expr input, Expr cos, Expr sin)
    {
        return CudaFusionRuleUtility.MakeFusionCall(
            $"{Name}_{Count++}",
            ModuleKind,
            root,
            (input, "input"),
            (cos, "cos"),
            (sin, "sin"));
    }
}

internal static class CudaFusionRuleUtility
{
    public static Pattern MaybeBoxed(Pattern input)
    {
        return IsAlt(input, IsCall((string?)null, IsBoxing(null!, _ => true, input)));
    }

    public static Pattern MaybeBoxed(string name, Pattern input)
    {
        return IsAlt(name, input, IsCall((string?)null, IsBoxing(null!, _ => true, input)));
    }

    public static TypePattern IsFloatLike()
    {
        return new TypePattern(
            type => type switch
            {
                TensorType tensorType => DataTypes.IsFloat(tensorType.DType),
                DistributedType distributedType => DataTypes.IsFloat(distributedType.TensorType.DType),
                _ => false,
            },
            "IsFloatLike");
    }

    public static TypePattern IsScalarLike()
    {
        return new TypePattern(
            type => type switch
            {
                TensorType tensorType => tensorType.IsScalar,
                DistributedType distributedType => distributedType.TensorType.IsScalar,
                _ => false,
            },
            "IsScalarLike");
    }

    public static Call? MakeFusionCall(string fusionName, string moduleKind, Call root, params (Expr Expr, string Name)[] inputs)
    {
        var parameters = new List<Expr>();
        var vars = new List<Var>();
        var varMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance);

        foreach (var (expr, name) in inputs)
        {
            if (varMap.ContainsKey(expr))
            {
                continue;
            }

            var @var = new Var(name, expr.CheckedType!);
            varMap.Add(expr, @var);
            vars.Add(@var);
            parameters.Add(expr);
        }

        var clonedRoot = new CudaFusionBodyCloner(varMap).Clone(root, default);
        var fusion = new Fusion(fusionName, moduleKind, clonedRoot, vars.ToArray());
        return new Call(fusion, parameters.ToArray());
    }

    public static bool IsDefaultSwishBeta(float beta) => global::System.Math.Abs(beta - 1.0f) <= 1e-6f;

    public static bool TryGetEquivalentInput(Expr lhs, Expr rhs, out Expr input)
    {
        input = lhs;
        return ReferenceEquals(Unbox(lhs), Unbox(rhs)) && Equals(lhs.CheckedType, rhs.CheckedType);
    }

    public static string EncodeLayerNormName(string baseName, int count, LayerNorm layerNorm)
    {
        var epsilon = layerNorm.Epsilon.ToString("R", CultureInfo.InvariantCulture);
        return $"{baseName}_a{layerNorm.Axis}_e{epsilon}_m{(layerNorm.UseMean ? 1 : 0)}_{count}";
    }

    public static string EncodeLayerNormTransposeName(string baseName, int count, LayerNorm layerNorm, IReadOnlyList<int> perm)
    {
        var permText = string.Join("_", perm);
        var epsilon = layerNorm.Epsilon.ToString("R", CultureInfo.InvariantCulture);
        return $"{baseName}_a{layerNorm.Axis}_e{epsilon}_m{(layerNorm.UseMean ? 1 : 0)}_p{permText}_{count}";
    }

    private static Expr Unbox(Expr expr)
    {
        while (expr is Call { Target: IR.Distributed.Boxing } call &&
               call.Arguments.Length == 1 &&
               call.Arguments[0] is Expr input)
        {
            expr = input;
        }

        return expr;
    }

    private sealed class CudaFusionBodyCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<Expr, Var> _varMap;

        public CudaFusionBodyCloner(IReadOnlyDictionary<Expr, Var> varMap)
        {
            _varMap = varMap;
        }

        protected override BaseExpr VisitCall(Call expr, Unit context)
        {
            if (_varMap.TryGetValue(expr, out var newVar))
            {
                return newVar;
            }

            return base.VisitCall(expr, context);
        }

        protected override Expr VisitLeafVar(Var expr, Unit context)
        {
            return _varMap.TryGetValue(expr, out var newVar) ? newVar : expr;
        }
    }
}
