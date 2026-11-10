"""MyoSim anatomical arms with motor-only shoulder and elbow actuation.

The lower-body model remains the original ``myolegs22`` model. Each upper
limb is taken directly from MyoSim's arm model, including its bone meshes and
segment inertias, but is reduced to two degrees of freedom:

* shoulder elevation/flexion;
* elbow flexion.

All upper-limb muscle actuators, tendons, muscle-path sites, wrapping geoms,
contacts, and the remaining arm joints are removed. The deleted joints become
fixed connections, so the clavicle/scapula, forearm rotation, wrist, and finger
segments remain anatomical rigid bodies without adding control dimensions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import myo_sim
from myo_sim.build.compose import build_mirrored_left_arm_spec, build_right_arm_spec


TORSO_ATTACHMENT_BODY = "cervical_spine"
ARM_ATTACHMENT_POSITIONS = {
    "r": (0.044675, -0.01807, 0.025465),
    "l": (0.044675, -0.01807, -0.025465),
}
_MOTOR_RGBA = (0.08, 0.36, 0.95, 1.0)
_PHANTOM_BODY_STEMS = ("clavphant", "scapphant", "humphant", "humphant1")
_PHANTOM_MASS_KG = 1e-3
# The source arm uses these bodies only to preserve the OpenSim kinematic
# hierarchy.  Its 1 kg m^2 diagonal inertia is incompatible with a 1 g body and
# adds a large, artificial rotational load to the torso.  A 1e-6 kg m^2
# diagonal is the existing placeholder convention in the torso model and is
# equivalent to a roughly 5 cm radius, 1 g solid sphere: small, but non-zero.
_PHANTOM_DIAGINERTIA_KGM2 = 1e-6
# The arm specs are attached below ``cervical_spine`` with a fixed frame whose
# rotation maps the child body's local +z axis onto world -y.  ``myolegs22``
# uses x-z as its sagittal plane (pelvis translation is x, vertical is z), so
# this is the exact hinge axis normal to that plane.  The native shoulder
# ``shoulder_elv`` axis is an elevation/abduction axis; keeping it unchanged
# would make the arm leave the sagittal plane even though only one shoulder
# coordinate survived.
_SAGITTAL_HINGE_AXIS_LOCAL = (0.0, 0.0, 1.0)


def _delete_all(spec: mujoco.MjSpec, collection_name: str) -> None:
    """Delete every element in one mutable MjSpec collection."""
    for element in list(getattr(spec, collection_name)):
        spec.delete(element)


def _prepare_arm_spec(spec: mujoco.MjSpec, side: str) -> None:
    """Reduce one native MyoSim arm to two rigid-body motor joints."""
    shoulder_name = f"shoulder_elv_{side}"
    elbow_name = f"elbow_flexion_{side}"
    kept_joints = {shoulder_name, elbow_name}

    # Nothing from the original upper-limb muscle transmission survives in the
    # motor-only model. Removing sites as well as tendons ensures there are no
    # dormant upper-limb muscle paths in the exported MJCF.
    for collection in ("equalities", "actuators", "tendons", "sites", "excludes"):
        _delete_all(spec, collection)

    for joint in list(spec.joints):
        if joint.name not in kept_joints:
            spec.delete(joint)

    # The native model gives every 1 g kinematic placeholder an inertia of
    # 1 kg m^2.  Once most arm joints are removed, those placeholders become
    # part of the fixed shoulder mounting chain and the impossible inertia
    # contaminates torso dynamics.  Keep their small mass so MuJoCo still has
    # valid, explicit inertials, but regularize the inertia to the same scale
    # used by other placeholder bodies in myotorso.
    for stem in _PHANTOM_BODY_STEMS:
        body = spec.body(f"{stem}_{side}")
        if body is None:
            raise ValueError(f"MyoArm is missing expected placeholder body {stem}_{side!s}")
        body.mass = _PHANTOM_MASS_KG
        body.inertia[:] = _PHANTOM_DIAGINERTIA_KGM2

    # Convert both retained anatomical coordinates into strict planar hinges.
    # At the neutral pose the attached arm bodies have local +z == world -y,
    # so both shoulders and elbows rotate only in the world x-z sagittal plane.
    # The same world-axis sign on both sides also makes positive flexion have a
    # consistent meaning for left/right policy mirroring.
    spec.joint(shoulder_name).axis[:] = _SAGITTAL_HINGE_AXIS_LOCAL
    spec.joint(elbow_name).axis[:] = _SAGITTAL_HINGE_AXIS_LOCAL

    # Retain only anatomical bone meshes. Capsule collision geometry and
    # muscle wrapping surfaces are unnecessary here; contacts are disabled for
    # the arms in this first locomotion-focused version. Segment mass and
    # inertia live on the bodies and are therefore preserved.
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
        else:
            geom.contype = 0
            geom.conaffinity = 0

    for joint_name, radius, half_length in (
        (shoulder_name, 0.038, 0.027),
        (elbow_name, 0.032, 0.022),
    ):
        joint = spec.joint(joint_name)
        axis = np.asarray(joint.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        endpoints = np.concatenate((-half_length * axis, half_length * axis))
        label = "shoulder" if joint_name == shoulder_name else "elbow"
        joint.parent.add_geom(
            name=f"{label}_motor_housing_{side}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, half_length, 0.0],
            fromto=endpoints,
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=2,
            rgba=_MOTOR_RGBA,
        )


def _strip_element_prefixes(spec: mujoco.MjSpec, prefix: str) -> None:
    """Restore readable anatomical names after collision-safe MjSpec attach."""
    for collection in ("bodies", "joints", "geoms", "frames"):
        for element in getattr(spec, collection):
            if element.name.startswith(prefix):
                element.name = element.name[len(prefix) :]


def _add_motor(spec: mujoco.MjSpec, name: str, target: str, gear: float) -> None:
    actuator = spec.add_actuator(
        name=name,
        target=target,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    actuator.set_to_motor()
    actuator.gear[0] = gear
    actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    actuator.ctrlrange[:] = (-1.0, 1.0)
    actuator.group = 2


def _absolutize_arm_assets(root: ET.Element) -> None:
    """Resolve relative arm mesh paths emitted by ``MjSpec.to_xml``."""
    models_dir = Path(myo_sim.__file__).resolve().parent / "models"
    for element in root.iter():
        file_name = element.get("file")
        if file_name and not Path(file_name).is_absolute():
            element.set("file", str((models_dir / file_name).resolve()))


def _joint_spans(model: mujoco.MjModel, address_field: str, total: int) -> dict[str, slice]:
    """Return each named joint's qpos or qvel slice in a compiled model."""
    addresses = getattr(model, address_field)
    spans: dict[str, slice] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            continue
        start = int(addresses[joint_id])
        stop = int(addresses[joint_id + 1]) if joint_id + 1 < model.njnt else total
        spans[name] = slice(start, stop)
    return spans


def _format_vector(values: np.ndarray) -> str:
    """Serialize a MuJoCo vector without adding insignificant noise."""
    return " ".join(f"{float(value):.15g}" for value in values)


def _remap_keyframes_by_joint(
    source_model: mujoco.MjModel,
    target_root: ET.Element,
) -> None:
    """Preserve keyframes semantically when attached joints change qpos order.

    ``MjSpec.attach`` retains the old keyframe vector and appends zeroes, but it
    inserts the arm joints near their parent body.  Consequently, old hip and
    knee values are otherwise interpreted as shoulder and elbow values.  Copy
    qpos/qvel by joint name instead; new arm coordinates remain at qpos0 with
    zero velocity.
    """
    key_elements = target_root.findall("./keyframe/key")
    if not key_elements:
        return

    target_model = mujoco.MjModel.from_xml_string(
        ET.tostring(target_root, encoding="unicode")
    )
    source_qpos = _joint_spans(source_model, "jnt_qposadr", source_model.nq)
    target_qpos = _joint_spans(target_model, "jnt_qposadr", target_model.nq)
    source_qvel = _joint_spans(source_model, "jnt_dofadr", source_model.nv)
    target_qvel = _joint_spans(target_model, "jnt_dofadr", target_model.nv)
    source_keys = {
        mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_KEY, key_id): key_id
        for key_id in range(source_model.nkey)
    }

    for target_key_id, element in enumerate(key_elements):
        name = element.get("name")
        source_key_id = source_keys.get(name, target_key_id)
        if source_key_id >= source_model.nkey:
            raise ValueError(f"Cannot remap unknown keyframe {name!r}")

        remapped_qpos = target_model.qpos0.copy()
        remapped_qvel = np.zeros(target_model.nv, dtype=float)
        for joint_name in source_qpos.keys() & target_qpos.keys():
            old_qpos, new_qpos = source_qpos[joint_name], target_qpos[joint_name]
            old_qvel, new_qvel = source_qvel[joint_name], target_qvel[joint_name]
            if old_qpos.stop - old_qpos.start != new_qpos.stop - new_qpos.start:
                raise ValueError(f"qpos width changed for joint {joint_name!r}")
            if old_qvel.stop - old_qvel.start != new_qvel.stop - new_qvel.start:
                raise ValueError(f"qvel width changed for joint {joint_name!r}")
            remapped_qpos[new_qpos] = source_model.key_qpos[source_key_id, old_qpos]
            remapped_qvel[new_qvel] = source_model.key_qvel[source_key_id, old_qvel]

        element.set("qpos", _format_vector(remapped_qpos))
        element.set("qvel", _format_vector(remapped_qvel))


def inject_motor_arms(root: ET.Element) -> None:
    """Attach two native anatomical, motor-only MyoSim arms in-place."""
    source_xml = ET.tostring(root, encoding="unicode")
    source_model = mujoco.MjModel.from_xml_string(source_xml)
    base = mujoco.MjSpec.from_string(source_xml)
    torso = base.body(TORSO_ATTACHMENT_BODY)
    if torso is None:
        raise ValueError(
            f"Cannot add anatomical arms: model has no body named {TORSO_ATTACHMENT_BODY!r}."
        )
    if base.joint("motor_shoulder_r") is not None:
        raise ValueError("Cannot add anatomical arms twice: motor_shoulder_r already exists")

    builders = {
        "r": build_right_arm_spec,
        "l": build_mirrored_left_arm_spec,
    }
    for side, builder in builders.items():
        arm = builder()
        _prepare_arm_spec(arm, side)

        # A prefix gives each attached spec its own default-class and asset
        # namespace. Readable body/joint/geom names are restored afterwards;
        # prefixed assets stay internal and avoid right/left name collisions.
        prefix = f"motor_arm_{side}__"
        frame = torso.add_frame(
            name=f"motor_arm_attach_{side}",
            pos=ARM_ATTACHMENT_POSITIONS[side],
        )
        base.attach(arm, frame=frame, prefix=prefix)
        _strip_element_prefixes(base, prefix)

        native_shoulder = f"shoulder_elv_{side}"
        native_elbow = f"elbow_flexion_{side}"
        shoulder = base.joint(native_shoulder)
        elbow = base.joint(native_elbow)
        shoulder.name = f"motor_shoulder_{side}"
        elbow.name = f"motor_elbow_{side}"

        # Zero is the anatomical hanging-arm pose.  Permit 60 degrees of
        # extension and 90 degrees of flexion for natural planar arm swing.
        shoulder.range[:] = (-np.pi / 3.0, np.pi / 2.0)
        shoulder.damping = 1.5
        shoulder.frictionloss = 0.05
        shoulder.armature = 0.03
        elbow.range[:] = (0.0, 2.269)
        elbow.damping = 1.0
        elbow.frictionloss = 0.04
        elbow.armature = 0.02

        _add_motor(base, shoulder.name, shoulder.name, gear=20.0)
        _add_motor(base, elbow.name, elbow.name, gear=12.0)

    rebuilt = ET.fromstring(base.to_xml())
    _absolutize_arm_assets(rebuilt)
    _remap_keyframes_by_joint(source_model, rebuilt)

    # Keep the public API in-place so the shared composition pipeline can apply
    # this transform after terrain seating without special-case serialization.
    root.clear()
    root.tag = rebuilt.tag
    root.attrib.update(rebuilt.attrib)
    root.text = rebuilt.text
    root.extend(list(rebuilt))
