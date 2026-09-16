# Cross-problem skill harness

5 general skills, 11 topic skills, 1003 tokens total.

Compiled over 144 problems from 209 curator decisions (89 accepted, 120 rejected).

## General skills

### `sk_0010` when-multiple-conditions-or-constraints-are-involved-in

*Selected 12x, 2 of those solved. 59 tokens. Evidence: p_100, p_101, p_102, p_103, p_136, p_16, p_17, p_21, p_23, p_33, p_34, p_35, p_38, p_40, p_41, p_42, p_44, p_45, p_96, p_98, p_99.*

**When to use.** When multiple constraints affect possible outcomes in a probability problem.

**Strategy.** - Enumerate all valid configurations respecting constraints.  
- Calculate probabilities for each case systematically.  
- Sum probabilities weighted by their occurrence.

**Avoid.** Overlooking distinct cases due to symmetry or constraint overlap.

### `sk_0013` when-multiple-constraints-must-be-satisfied-simultaneously-o

*Selected 31x, 5 of those solved. 55 tokens. Evidence: p_51, p_52, p_53, p_55, p_56, p_57, p_58, p_59, p_60, p_62, p_63, p_65, p_66, p_68, p_69, p_70, p_71, p_80, p_81, p_82, p_85, p_86, p_87.*

**When to use.** When deriving conclusions from patterns or shared values.

**Strategy.** - Verify assumptions with multiple examples.  
- Test edge cases to expose hidden constraints.  
- Use algebraic or logical methods to confirm relationships.

**Avoid.** Assuming relationships hold without systematic validation.

### `sk_0006` when-solving-a-problem-ensure-all-possible-cases

*Selected 118x, 27 of those solved. 77 tokens. Evidence: p_0, p_112, p_113, p_114, p_115, p_116, p_117, p_118, p_119, p_122, p_123, p_124, p_125, p_126, p_127, p_129, p_130, p_2, p_22, p_24, p_25, p_26, p_27, p_28, p_29, p_3, p_30, p_40, p_41, p_42, p_44, p_45, p_48, p_49, p_5, p_51, p_52, p_53, p_54, p_55, p_6, p_7, p_73, p_74, p_79.*

**When to use.** When a circle is tangent to multiple sides of a figure and intersects a diagonal.

**Strategy.** - Use symmetry and tangency to locate the circle's center.  
- Leverage coordinate geometry to model relationships.  
- Combine segment lengths with intersection points to find key coordinates.

**Avoid.** Assuming tangency implies equal distances without verifying with coordinates.

### `sk_0009` when-solving-a-problem-with-multiple-conditions-or

*Selected 11x, 3 of those solved. 51 tokens. Evidence: p_12, p_13, p_131, p_133, p_134, p_14, p_40, p_41, p_42, p_44, p_45, p_8, p_9.*

**When to use.** When solving systems with multiple interdependent equations.

**Strategy.** - Avoid symmetry assumptions without verification.  
- Check consistency across all equations.  
- Use substitution to reveal hidden constraints.

**Avoid.** Assuming equal variables without validating against all equations.

### `sk_0014` when-solving-problems-with-multiple-steps-or-positional

*Selected 26x, 6 of those solved. 58 tokens. Evidence: p_112, p_113, p_114, p_115, p_116, p_117, p_118, p_119, p_128, p_129, p_130, p_131, p_132, p_133, p_134, p_136, p_138, p_139, p_141, p_142, p_143, p_49, p_54, p_88, p_89, p_90, p_92, p_93, p_95.*

**When to use.** When multiple constraints affect the validity of an arrangement.

**Strategy.** - Systematically check all combinations under constraints.  
- Validate each condition explicitly.  
- Avoid assumptions; verify all required properties.

**Avoid.** Failing to ensure all conditions are met in combinatorial problems.

## Topic skills

### Topic: adjacent_grouping_counting

### `sk_0002` when-counting-favorable-outcomes-involving-adjacent-elements

*Selected 9x, 2 of those solved. 63 tokens. Evidence: p_13, p_2.*

**When to use.** When counting paths with adjacency restrictions in grids or sequences.

**Strategy.** - Model transitions between states based on adjacency rules.  
- Use DP to track valid configurations.  
- Handle edge cases like boundaries or stopping points explicitly.

**Avoid.** Ignoring boundary conditions or misrepresenting state transitions.

### Topic: case_analysis

### `sk_0007` when-solving-combinatorial-problems-with-multiple-conditions

*Selected 47x, 7 of those solved. 65 tokens. Evidence: p_17, p_22, p_8.*

**When to use.** When solving combinatorial problems with multiple conditions

**Strategy.** - Enumerate all valid move combinations systematically.  
- Track intermediate results to avoid double-counting or missing cases.  
- Verify edge cases and ensure all constraints are satisfied.

**Avoid.** Assuming a single path type or neglecting step size variations.

### Topic: combinatorial_constraints

### `sk_0012` when-solving-problems-with-multiple-constraints-on-variable

*Selected 31x, 6 of those solved. 58 tokens. Evidence: p_115, p_132, p_136, p_139, p_29, p_33, p_38, p_56, p_63, p_66, p_73, p_79, p_87, p_96.*

**When to use.** When multiple constraints affect the validity of an arrangement.

**Strategy.** - Systematically check all combinations under constraints.  
- Validate each condition explicitly.  
- Avoid assumptions; verify all required properties.

**Avoid.** Failing to ensure all conditions are met in combinatorial problems.

### Topic: combinatorial_reasoning

### `sk_0011` when-solving-problems-with-complex-combinatorial-conditions

*Selected 16x, 2 of those solved. 63 tokens. Evidence: p_117, p_143, p_25, p_34, p_40, p_41, p_42, p_44, p_51, p_65, p_74.*

**When to use.** When encountering nested binomial coefficients or complex summations.

**Strategy.** - Derive closed-form expressions for nested binomial terms.  
- Simplify before computing to reduce complexity.  
- Use modular arithmetic at each step to handle large numbers.

**Avoid.** Leaving summations unresolved or providing incomplete answers.

### Topic: coordinate_geometry_setup

### `sk_0003` when-solving-geometry-problems-with-multiple-segments-and

*Selected 28x, 6 of those solved. 64 tokens. Evidence: p_12, p_3, p_81, p_95.*

**When to use.** When setting up coordinates for a configuration with multiple distance constraints in 3D space.

**Strategy.** - Use symmetry to simplify coordinate assignments.  
- Subtract equations to eliminate variables and find relationships.  
- Substitute systematically to reduce complexity.

**Avoid.** Assuming coordinates without verifying consistency with all constraints.

### Topic: equiangular_hexagon_properties

### `sk_0005` when-solving-for-inradius-or-other-properties-of

*Selected 1x, 0 of those solved. 93 tokens. Evidence: p_18, p_7.*

**When to use.** When calculating area of an equiangular hexagon with alternating side lengths.

**Strategy.** - Use $ \frac{1}{2}ab\sin(\theta) $ for triangles with two sides and included angle.  
- Recognize consistent internal angles in equiangular polygons.  
- Multiply individual triangle areas by number of triangles.

**Avoid.** Assuming all triangles have same side lengths or angles without verification.

### Topic: factorization_conditions

### `sk_0001` when-solving-problems-involving-polynomial-factorization-wit

*Selected 18x, 9 of those solved. 63 tokens. Evidence: p_0.*

**When to use.** when solving problems involving polynomial factorization with integer coefficients

**Strategy.** - Identify necessary conditions for factorization.  
- Translate constraints into equations involving roots.  
- Count valid root pairs that satisfy both sum and product conditions.

**Avoid.** providing answers without showing the logical steps or reasoning process.

### Topic: geometric_constraints

### `sk_0015` when-solving-a-geometric-problem-with-multiple-distance

*Selected 15x, 2 of those solved. 62 tokens. Evidence: p_104, p_106, p_114, p_122, p_124, p_141, p_98.*

**When to use.** When tangency and triangle formation are involved with potential symmetry.

**Strategy.** - Verify geometric assumptions with diagrams.  
- Check if triangle properties align with given conditions.  
- Avoid assuming symmetry without proof.

**Avoid.** Assuming a triangle is equilateral based on tangent properties without justification.

### Topic: isosceles_triangle_counting

### `sk_0004` when-counting-isosceles-triangles-in-geometric-configuration

*Selected 0x, 0 of those solved. 58 tokens. Evidence: p_6.*

**When to use.** when counting isosceles triangles in geometric configurations

**Strategy.** - Systematically categorize cases based on vertex distribution.  
- Use symmetry and known distances in regular polygons.  
- Verify all cases to avoid missing configurations.

**Avoid.** failing to consider different vertex distributions across bases.

### Topic: logical_reasoning

### `sk_0008` when-solving-problems-with-multiple-conditions-and-constrain

*Selected 74x, 20 of those solved. 60 tokens. Evidence: p_133, p_138, p_14, p_16, p_21, p_23, p_30, p_35, p_54, p_59, p_85, p_89.*

**When to use.** When solving a problem with multiple conditions or constraints

**Strategy.** - Check domain validity of expressions before finalizing answers.  
- Validate intermediate steps with numerical examples.  
- Ensure all logarithmic and algebraic properties are correctly applied.

**Avoid.** Assuming intermediate results are valid without verification.

### Topic: sequential_dependencies

### `sk_0016` when-solving-problems-with-recursive-or-step-based

*Selected 1x, 0 of those solved. 54 tokens. Evidence: p_119.*

**When to use.** When solving problems with recursive or step-based functions

**Strategy.** - Track function behavior across all relevant inputs.  
- Verify intermediate steps for consistency.  
- Confirm ratios match exact conditions.

**Avoid.** Assuming intermediate values are correct without verification.

