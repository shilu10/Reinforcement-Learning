��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58˔
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
 critic_network_19/dense_159/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" critic_network_19/dense_159/bias
�
4critic_network_19/dense_159/bias/Read/ReadVariableOpReadVariableOp critic_network_19/dense_159/bias*
_output_shapes
:*
dtype0
�
"critic_network_19/dense_159/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *3
shared_name$"critic_network_19/dense_159/kernel
�
6critic_network_19/dense_159/kernel/Read/ReadVariableOpReadVariableOp"critic_network_19/dense_159/kernel*
_output_shapes

: *
dtype0
�
 critic_network_19/dense_158/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" critic_network_19/dense_158/bias
�
4critic_network_19/dense_158/bias/Read/ReadVariableOpReadVariableOp critic_network_19/dense_158/bias*
_output_shapes
: *
dtype0
�
"critic_network_19/dense_158/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *3
shared_name$"critic_network_19/dense_158/kernel
�
6critic_network_19/dense_158/kernel/Read/ReadVariableOpReadVariableOp"critic_network_19/dense_158/kernel*
_output_shapes

:  *
dtype0
�
 critic_network_19/dense_157/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" critic_network_19/dense_157/bias
�
4critic_network_19/dense_157/bias/Read/ReadVariableOpReadVariableOp critic_network_19/dense_157/bias*
_output_shapes
: *
dtype0
�
"critic_network_19/dense_157/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *3
shared_name$"critic_network_19/dense_157/kernel
�
6critic_network_19/dense_157/kernel/Read/ReadVariableOpReadVariableOp"critic_network_19/dense_157/kernel*
_output_shapes

:@ *
dtype0
�
 critic_network_19/dense_156/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" critic_network_19/dense_156/bias
�
4critic_network_19/dense_156/bias/Read/ReadVariableOpReadVariableOp critic_network_19/dense_156/bias*
_output_shapes
:@*
dtype0
�
"critic_network_19/dense_156/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"critic_network_19/dense_156/kernel
�
6critic_network_19/dense_156/kernel/Read/ReadVariableOpReadVariableOp"critic_network_19/dense_156/kernel*
_output_shapes

:@*
dtype0
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
y
serving_default_args_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1"critic_network_19/dense_156/kernel critic_network_19/dense_156/bias"critic_network_19/dense_157/kernel critic_network_19/dense_157/bias"critic_network_19/dense_158/kernel critic_network_19/dense_158/bias"critic_network_19/dense_159/kernel critic_network_19/dense_159/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_56447636

NoOpNoOp
� 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*� 
value� B�  B� 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
out
	optimizer
loss

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
O
6
_variables
7_iterations
8_learning_rate
9_update_step_xla*
* 

:serving_default* 
b\
VARIABLE_VALUE"critic_network_19/dense_156/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_19/dense_156/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"critic_network_19/dense_157/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_19/dense_157/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"critic_network_19/dense_158/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_19/dense_158/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"critic_network_19/dense_159/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE critic_network_19/dense_159/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 

0
1*

0
1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 

0
1*

0
1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 

70*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6critic_network_19/dense_156/kernel/Read/ReadVariableOp4critic_network_19/dense_156/bias/Read/ReadVariableOp6critic_network_19/dense_157/kernel/Read/ReadVariableOp4critic_network_19/dense_157/bias/Read/ReadVariableOp6critic_network_19/dense_158/kernel/Read/ReadVariableOp4critic_network_19/dense_158/bias/Read/ReadVariableOp6critic_network_19/dense_159/kernel/Read/ReadVariableOp4critic_network_19/dense_159/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__traced_save_56447825
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"critic_network_19/dense_156/kernel critic_network_19/dense_156/bias"critic_network_19/dense_157/kernel critic_network_19/dense_157/bias"critic_network_19/dense_158/kernel critic_network_19/dense_158/bias"critic_network_19/dense_159/kernel critic_network_19/dense_159/bias	iterationlearning_rate*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference__traced_restore_56447865��
�&
�
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447692	
state

action:
(dense_156_matmul_readvariableop_resource:@7
)dense_156_biasadd_readvariableop_resource:@:
(dense_157_matmul_readvariableop_resource:@ 7
)dense_157_biasadd_readvariableop_resource: :
(dense_158_matmul_readvariableop_resource:  7
)dense_158_biasadd_readvariableop_resource: :
(dense_159_matmul_readvariableop_resource: 7
)dense_159_biasadd_readvariableop_resource:
identity�� dense_156/BiasAdd/ReadVariableOp�dense_156/MatMul/ReadVariableOp� dense_157/BiasAdd/ReadVariableOp�dense_157/MatMul/ReadVariableOp� dense_158/BiasAdd/ReadVariableOp�dense_158/MatMul/ReadVariableOp� dense_159/BiasAdd/ReadVariableOp�dense_159/MatMul/ReadVariableOpM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense_156/MatMul/ReadVariableOpReadVariableOp(dense_156_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_156/MatMulMatMulconcat:output:0'dense_156/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_156/BiasAdd/ReadVariableOpReadVariableOp)dense_156_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_156/BiasAddBiasAdddense_156/MatMul:product:0(dense_156/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
dense_156/ReluReludense_156/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_157/MatMul/ReadVariableOpReadVariableOp(dense_157_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_157/MatMulMatMuldense_156/Relu:activations:0'dense_157/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_157/BiasAdd/ReadVariableOpReadVariableOp)dense_157_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_157/BiasAddBiasAdddense_157/MatMul:product:0(dense_157/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_157/ReluReludense_157/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_158/MatMul/ReadVariableOpReadVariableOp(dense_158_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_158/MatMulMatMuldense_157/Relu:activations:0'dense_158/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_158/BiasAdd/ReadVariableOpReadVariableOp)dense_158_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_158/BiasAddBiasAdddense_158/MatMul:product:0(dense_158/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_158/ReluReludense_158/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_159/MatMul/ReadVariableOpReadVariableOp(dense_159_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_159/MatMulMatMuldense_158/Relu:activations:0'dense_159/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_159/BiasAdd/ReadVariableOpReadVariableOp)dense_159_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_159/BiasAddBiasAdddense_159/MatMul:product:0(dense_159/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_159/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_156/BiasAdd/ReadVariableOp ^dense_156/MatMul/ReadVariableOp!^dense_157/BiasAdd/ReadVariableOp ^dense_157/MatMul/ReadVariableOp!^dense_158/BiasAdd/ReadVariableOp ^dense_158/MatMul/ReadVariableOp!^dense_159/BiasAdd/ReadVariableOp ^dense_159/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������: : : : : : : : 2D
 dense_156/BiasAdd/ReadVariableOp dense_156/BiasAdd/ReadVariableOp2B
dense_156/MatMul/ReadVariableOpdense_156/MatMul/ReadVariableOp2D
 dense_157/BiasAdd/ReadVariableOp dense_157/BiasAdd/ReadVariableOp2B
dense_157/MatMul/ReadVariableOpdense_157/MatMul/ReadVariableOp2D
 dense_158/BiasAdd/ReadVariableOp dense_158/BiasAdd/ReadVariableOp2B
dense_158/MatMul/ReadVariableOpdense_158/MatMul/ReadVariableOp2D
 dense_159/BiasAdd/ReadVariableOp dense_159/BiasAdd/ReadVariableOp2B
dense_159/MatMul/ReadVariableOpdense_159/MatMul/ReadVariableOp:N J
'
_output_shapes
:���������

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�

�
&__inference_signature_wrapper_56447636

args_0

args_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_56447472o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_1
�/
�
$__inference__traced_restore_56447865
file_prefixE
3assignvariableop_critic_network_19_dense_156_kernel:@A
3assignvariableop_1_critic_network_19_dense_156_bias:@G
5assignvariableop_2_critic_network_19_dense_157_kernel:@ A
3assignvariableop_3_critic_network_19_dense_157_bias: G
5assignvariableop_4_critic_network_19_dense_158_kernel:  A
3assignvariableop_5_critic_network_19_dense_158_bias: G
5assignvariableop_6_critic_network_19_dense_159_kernel: A
3assignvariableop_7_critic_network_19_dense_159_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: 
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp3assignvariableop_critic_network_19_dense_156_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp3assignvariableop_1_critic_network_19_dense_156_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_critic_network_19_dense_157_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp3assignvariableop_3_critic_network_19_dense_157_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_critic_network_19_dense_158_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp3assignvariableop_5_critic_network_19_dense_158_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp5assignvariableop_6_critic_network_19_dense_159_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp3assignvariableop_7_critic_network_19_dense_159_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
4__inference_critic_network_19_layer_call_fn_56447658	
state

action
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstateactionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�
�
,__inference_dense_158_layer_call_fn_56447741

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_158_layer_call_and_return_conditional_losses_56447528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
G__inference_dense_158_layer_call_and_return_conditional_losses_56447528

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
G__inference_dense_157_layer_call_and_return_conditional_losses_56447732

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
!__inference__traced_save_56447825
file_prefixA
=savev2_critic_network_19_dense_156_kernel_read_readvariableop?
;savev2_critic_network_19_dense_156_bias_read_readvariableopA
=savev2_critic_network_19_dense_157_kernel_read_readvariableop?
;savev2_critic_network_19_dense_157_bias_read_readvariableopA
=savev2_critic_network_19_dense_158_kernel_read_readvariableop?
;savev2_critic_network_19_dense_158_bias_read_readvariableopA
=savev2_critic_network_19_dense_159_kernel_read_readvariableop?
;savev2_critic_network_19_dense_159_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_critic_network_19_dense_156_kernel_read_readvariableop;savev2_critic_network_19_dense_156_bias_read_readvariableop=savev2_critic_network_19_dense_157_kernel_read_readvariableop;savev2_critic_network_19_dense_157_bias_read_readvariableop=savev2_critic_network_19_dense_158_kernel_read_readvariableop;savev2_critic_network_19_dense_158_bias_read_readvariableop=savev2_critic_network_19_dense_159_kernel_read_readvariableop;savev2_critic_network_19_dense_159_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*[
_input_shapesJ
H: :@:@:@ : :  : : :: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�

�
G__inference_dense_156_layer_call_and_return_conditional_losses_56447494

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_159_layer_call_fn_56447761

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_159_layer_call_and_return_conditional_losses_56447544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
G__inference_dense_159_layer_call_and_return_conditional_losses_56447544

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
G__inference_dense_158_layer_call_and_return_conditional_losses_56447752

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
G__inference_dense_157_layer_call_and_return_conditional_losses_56447511

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
G__inference_dense_156_layer_call_and_return_conditional_losses_56447712

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�
#__inference__wrapped_model_56447472

args_0

args_1L
:critic_network_19_dense_156_matmul_readvariableop_resource:@I
;critic_network_19_dense_156_biasadd_readvariableop_resource:@L
:critic_network_19_dense_157_matmul_readvariableop_resource:@ I
;critic_network_19_dense_157_biasadd_readvariableop_resource: L
:critic_network_19_dense_158_matmul_readvariableop_resource:  I
;critic_network_19_dense_158_biasadd_readvariableop_resource: L
:critic_network_19_dense_159_matmul_readvariableop_resource: I
;critic_network_19_dense_159_biasadd_readvariableop_resource:
identity��2critic_network_19/dense_156/BiasAdd/ReadVariableOp�1critic_network_19/dense_156/MatMul/ReadVariableOp�2critic_network_19/dense_157/BiasAdd/ReadVariableOp�1critic_network_19/dense_157/MatMul/ReadVariableOp�2critic_network_19/dense_158/BiasAdd/ReadVariableOp�1critic_network_19/dense_158/MatMul/ReadVariableOp�2critic_network_19/dense_159/BiasAdd/ReadVariableOp�1critic_network_19/dense_159/MatMul/ReadVariableOp_
critic_network_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
critic_network_19/concatConcatV2args_0args_1&critic_network_19/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
1critic_network_19/dense_156/MatMul/ReadVariableOpReadVariableOp:critic_network_19_dense_156_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
"critic_network_19/dense_156/MatMulMatMul!critic_network_19/concat:output:09critic_network_19/dense_156/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2critic_network_19/dense_156/BiasAdd/ReadVariableOpReadVariableOp;critic_network_19_dense_156_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#critic_network_19/dense_156/BiasAddBiasAdd,critic_network_19/dense_156/MatMul:product:0:critic_network_19/dense_156/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 critic_network_19/dense_156/ReluRelu,critic_network_19/dense_156/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1critic_network_19/dense_157/MatMul/ReadVariableOpReadVariableOp:critic_network_19_dense_157_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
"critic_network_19/dense_157/MatMulMatMul.critic_network_19/dense_156/Relu:activations:09critic_network_19/dense_157/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
2critic_network_19/dense_157/BiasAdd/ReadVariableOpReadVariableOp;critic_network_19_dense_157_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#critic_network_19/dense_157/BiasAddBiasAdd,critic_network_19/dense_157/MatMul:product:0:critic_network_19/dense_157/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 critic_network_19/dense_157/ReluRelu,critic_network_19/dense_157/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1critic_network_19/dense_158/MatMul/ReadVariableOpReadVariableOp:critic_network_19_dense_158_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
"critic_network_19/dense_158/MatMulMatMul.critic_network_19/dense_157/Relu:activations:09critic_network_19/dense_158/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
2critic_network_19/dense_158/BiasAdd/ReadVariableOpReadVariableOp;critic_network_19_dense_158_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#critic_network_19/dense_158/BiasAddBiasAdd,critic_network_19/dense_158/MatMul:product:0:critic_network_19/dense_158/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 critic_network_19/dense_158/ReluRelu,critic_network_19/dense_158/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1critic_network_19/dense_159/MatMul/ReadVariableOpReadVariableOp:critic_network_19_dense_159_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"critic_network_19/dense_159/MatMulMatMul.critic_network_19/dense_158/Relu:activations:09critic_network_19/dense_159/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2critic_network_19/dense_159/BiasAdd/ReadVariableOpReadVariableOp;critic_network_19_dense_159_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#critic_network_19/dense_159/BiasAddBiasAdd,critic_network_19/dense_159/MatMul:product:0:critic_network_19/dense_159/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,critic_network_19/dense_159/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^critic_network_19/dense_156/BiasAdd/ReadVariableOp2^critic_network_19/dense_156/MatMul/ReadVariableOp3^critic_network_19/dense_157/BiasAdd/ReadVariableOp2^critic_network_19/dense_157/MatMul/ReadVariableOp3^critic_network_19/dense_158/BiasAdd/ReadVariableOp2^critic_network_19/dense_158/MatMul/ReadVariableOp3^critic_network_19/dense_159/BiasAdd/ReadVariableOp2^critic_network_19/dense_159/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������: : : : : : : : 2h
2critic_network_19/dense_156/BiasAdd/ReadVariableOp2critic_network_19/dense_156/BiasAdd/ReadVariableOp2f
1critic_network_19/dense_156/MatMul/ReadVariableOp1critic_network_19/dense_156/MatMul/ReadVariableOp2h
2critic_network_19/dense_157/BiasAdd/ReadVariableOp2critic_network_19/dense_157/BiasAdd/ReadVariableOp2f
1critic_network_19/dense_157/MatMul/ReadVariableOp1critic_network_19/dense_157/MatMul/ReadVariableOp2h
2critic_network_19/dense_158/BiasAdd/ReadVariableOp2critic_network_19/dense_158/BiasAdd/ReadVariableOp2f
1critic_network_19/dense_158/MatMul/ReadVariableOp1critic_network_19/dense_158/MatMul/ReadVariableOp2h
2critic_network_19/dense_159/BiasAdd/ReadVariableOp2critic_network_19/dense_159/BiasAdd/ReadVariableOp2f
1critic_network_19/dense_159/MatMul/ReadVariableOp1critic_network_19/dense_159/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:OK
'
_output_shapes
:���������
 
_user_specified_nameargs_1
�
�
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447551	
state

action$
dense_156_56447495:@ 
dense_156_56447497:@$
dense_157_56447512:@  
dense_157_56447514: $
dense_158_56447529:   
dense_158_56447531: $
dense_159_56447545:  
dense_159_56447547:
identity��!dense_156/StatefulPartitionedCall�!dense_157/StatefulPartitionedCall�!dense_158/StatefulPartitionedCall�!dense_159/StatefulPartitionedCallM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :r
concatConcatV2stateactionconcat/axis:output:0*
N*
T0*'
_output_shapes
:����������
!dense_156/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_156_56447495dense_156_56447497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_56447494�
!dense_157/StatefulPartitionedCallStatefulPartitionedCall*dense_156/StatefulPartitionedCall:output:0dense_157_56447512dense_157_56447514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_56447511�
!dense_158/StatefulPartitionedCallStatefulPartitionedCall*dense_157/StatefulPartitionedCall:output:0dense_158_56447529dense_158_56447531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_158_layer_call_and_return_conditional_losses_56447528�
!dense_159/StatefulPartitionedCallStatefulPartitionedCall*dense_158/StatefulPartitionedCall:output:0dense_159_56447545dense_159_56447547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_159_layer_call_and_return_conditional_losses_56447544y
IdentityIdentity*dense_159/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_156/StatefulPartitionedCall"^dense_157/StatefulPartitionedCall"^dense_158/StatefulPartitionedCall"^dense_159/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������:���������: : : : : : : : 2F
!dense_156/StatefulPartitionedCall!dense_156/StatefulPartitionedCall2F
!dense_157/StatefulPartitionedCall!dense_157/StatefulPartitionedCall2F
!dense_158/StatefulPartitionedCall!dense_158/StatefulPartitionedCall2F
!dense_159/StatefulPartitionedCall!dense_159/StatefulPartitionedCall:N J
'
_output_shapes
:���������

_user_specified_namestate:OK
'
_output_shapes
:���������
 
_user_specified_nameaction
�
�
,__inference_dense_157_layer_call_fn_56447721

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_157_layer_call_and_return_conditional_losses_56447511o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_dense_156_layer_call_fn_56447701

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_dense_156_layer_call_and_return_conditional_losses_56447494o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_dense_159_layer_call_and_return_conditional_losses_56447771

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0���������
9
args_1/
serving_default_args_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�b
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
out
	optimizer
loss

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
4__inference_critic_network_19_layer_call_fn_56447658�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447692�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
#__inference__wrapped_model_56447472args_0args_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
j
6
_variables
7_iterations
8_learning_rate
9_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
:serving_default"
signature_map
4:2@2"critic_network_19/dense_156/kernel
.:,@2 critic_network_19/dense_156/bias
4:2@ 2"critic_network_19/dense_157/kernel
.:, 2 critic_network_19/dense_157/bias
4:2  2"critic_network_19/dense_158/kernel
.:, 2 critic_network_19/dense_158/bias
4:2 2"critic_network_19/dense_159/kernel
.:,2 critic_network_19/dense_159/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_critic_network_19_layer_call_fn_56447658stateaction"�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447692stateaction"�
���
FullArgSpec&
args�
jself
jstate
jaction
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
,__inference_dense_156_layer_call_fn_56447701�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
�
Atrace_02�
G__inference_dense_156_layer_call_and_return_conditional_losses_56447712�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
,__inference_dense_157_layer_call_fn_56447721�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
G__inference_dense_157_layer_call_and_return_conditional_losses_56447732�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
,__inference_dense_158_layer_call_fn_56447741�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
G__inference_dense_158_layer_call_and_return_conditional_losses_56447752�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
Utrace_02�
,__inference_dense_159_layer_call_fn_56447761�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
�
Vtrace_02�
G__inference_dense_159_layer_call_and_return_conditional_losses_56447771�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
'
70"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_56447636args_0args_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_156_layer_call_fn_56447701inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_156_layer_call_and_return_conditional_losses_56447712inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_157_layer_call_fn_56447721inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_157_layer_call_and_return_conditional_losses_56447732inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_158_layer_call_fn_56447741inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_158_layer_call_and_return_conditional_losses_56447752inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_159_layer_call_fn_56447761inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_159_layer_call_and_return_conditional_losses_56447771inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_56447472�Q�N
G�D
 �
args_0���������
 �
args_1���������
� "3�0
.
output_1"�
output_1����������
O__inference_critic_network_19_layer_call_and_return_conditional_losses_56447692�P�M
F�C
�
state���������
 �
action���������
� ",�)
"�
tensor_0���������
� �
4__inference_critic_network_19_layer_call_fn_56447658P�M
F�C
�
state���������
 �
action���������
� "!�
unknown����������
G__inference_dense_156_layer_call_and_return_conditional_losses_56447712c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������@
� �
,__inference_dense_156_layer_call_fn_56447701X/�,
%�"
 �
inputs���������
� "!�
unknown���������@�
G__inference_dense_157_layer_call_and_return_conditional_losses_56447732c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
,__inference_dense_157_layer_call_fn_56447721X/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
G__inference_dense_158_layer_call_and_return_conditional_losses_56447752c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
,__inference_dense_158_layer_call_fn_56447741X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
G__inference_dense_159_layer_call_and_return_conditional_losses_56447771c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
,__inference_dense_159_layer_call_fn_56447761X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
&__inference_signature_wrapper_56447636�e�b
� 
[�X
*
args_0 �
args_0���������
*
args_1 �
args_1���������"3�0
.
output_1"�
output_1���������