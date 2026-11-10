from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myoassist_utils.compose import compose_motor_arms_model, compose_msk_model


@pytest.fixture(scope="module")
def motor_arms_model():
    xml = compose_motor_arms_model("myolegs22")
    return mujoco.MjModel.from_xml_string(xml)


@pytest.fixture(scope="module")
def base_model():
    return mujoco.MjModel.from_xml_string(compose_msk_model("myolegs22"))


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]


def test_motor_arms_keep_22_leg_muscles_and_add_only_four_motors(motor_arms_model):
    model = motor_arms_model
    is_muscle = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE

    assert model.nu == 26
    assert int(is_muscle.sum()) == 22
    assert int((~is_muscle).sum()) == 4

    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    assert actuator_names[-4:] == [
        "motor_shoulder_r",
        "motor_elbow_r",
        "motor_shoulder_l",
        "motor_elbow_l",
    ]


def test_motor_arms_have_only_requested_shoulder_and_elbow_joints(motor_arms_model):
    model = motor_arms_model
    joint_names = set(_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt))
    body_names = set(_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody))

    assert {
        "motor_shoulder_r",
        "motor_elbow_r",
        "motor_shoulder_l",
        "motor_elbow_l",
    } <= joint_names
    assert {"clavicle_r", "scapula_r", "humerus_r", "ulna_r", "radius_r"} <= body_names
    assert {"clavicle_l", "scapula_l", "humerus_l", "ulna_l", "radius_l"} <= body_names
    assert not any(name.startswith("motor_upper_arm") for name in body_names)
    assert not any(name.startswith("motor_wrist") for name in joint_names)
    assert not any(name.startswith("motor_finger") for name in joint_names)
    assert model.njnt == 43


def test_motor_arm_joints_are_strict_sagittal_hinges(motor_arms_model):
    model = motor_arms_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    for side in ("r", "l"):
        shoulder = model.joint(f"motor_shoulder_{side}")
        elbow = model.joint(f"motor_elbow_{side}")

        np.testing.assert_allclose(
            model.jnt_range[shoulder.id],
            [-np.pi / 3.0, np.pi / 2.0],
            atol=5e-6,
        )
        np.testing.assert_allclose(
            model.jnt_range[elbow.id],
            [0.0, 2.269],
            atol=1e-6,
        )

        for joint in (shoulder, elbow):
            body_id = model.jnt_bodyid[joint.id]
            body_rotation = data.xmat[body_id].reshape(3, 3)
            world_axis = body_rotation @ model.jnt_axis[joint.id]
            # myolegs22's sagittal plane is world x-z, hence its normal is y.
            np.testing.assert_allclose(np.abs(world_axis), [0.0, 1.0, 0.0], atol=1e-6)


def test_motor_arm_geoms_start_noncolliding(motor_arms_model):
    model = motor_arms_model
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(("shoulder_motor_", "elbow_motor_")):
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0


def test_upper_limbs_have_no_muscle_paths_or_actuators(motor_arms_model):
    model = motor_arms_model
    tendon_names = _names(model, mujoco.mjtObj.mjOBJ_TENDON, model.ntendon)
    site_names = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)

    # All surviving tendons/sites belong to the original myolegs22 model.
    assert not any(name and name.startswith("motor_arm_") for name in tendon_names)
    assert not any(name and name.startswith("motor_arm_") for name in site_names)
    assert model.ntendon == 22
    assert model.nsite == 92
    assert model.nu == 22 + 4


def _joint_span(model: mujoco.MjModel, joint_id: int, *, qpos: bool) -> slice:
    addresses = model.jnt_qposadr if qpos else model.jnt_dofadr
    total = model.nq if qpos else model.nv
    start = int(addresses[joint_id])
    stop = int(addresses[joint_id + 1]) if joint_id + 1 < model.njnt else total
    return slice(start, stop)


def test_keyframes_preserve_all_base_joint_values_and_zero_new_arms(
    base_model,
    motor_arms_model,
):
    base_keys = {base_model.key(i).name: i for i in range(base_model.nkey)}

    for target_key_id in range(motor_arms_model.nkey):
        key_name = motor_arms_model.key(target_key_id).name
        source_key_id = base_keys[key_name]
        for source_joint_id in range(base_model.njnt):
            joint_name = base_model.joint(source_joint_id).name
            target_joint_id = motor_arms_model.joint(joint_name).id
            source_qpos = _joint_span(base_model, source_joint_id, qpos=True)
            target_qpos = _joint_span(motor_arms_model, target_joint_id, qpos=True)
            source_qvel = _joint_span(base_model, source_joint_id, qpos=False)
            target_qvel = _joint_span(motor_arms_model, target_joint_id, qpos=False)
            np.testing.assert_allclose(
                motor_arms_model.key_qpos[target_key_id, target_qpos],
                base_model.key_qpos[source_key_id, source_qpos],
            )
            np.testing.assert_allclose(
                motor_arms_model.key_qvel[target_key_id, target_qvel],
                base_model.key_qvel[source_key_id, source_qvel],
            )

        for joint_name in (
            "motor_shoulder_r",
            "motor_elbow_r",
            "motor_shoulder_l",
            "motor_elbow_l",
        ):
            joint = motor_arms_model.joint(joint_name)
            value = motor_arms_model.key_qpos[target_key_id, joint.qposadr]
            assert value == pytest.approx(0.0)
            assert motor_arms_model.jnt_range[joint.id, 0] <= value
            assert value <= motor_arms_model.jnt_range[joint.id, 1]


def test_phantom_bodies_have_small_nonzero_placeholder_inertia(motor_arms_model):
    for side in ("r", "l"):
        for stem in ("clavphant", "scapphant", "humphant", "humphant1"):
            body = motor_arms_model.body(f"{stem}_{side}")
            assert motor_arms_model.body_mass[body.id] == pytest.approx(1e-3)
            np.testing.assert_allclose(
                motor_arms_model.body_inertia[body.id],
                [1e-6, 1e-6, 1e-6],
                rtol=0.0,
                atol=1e-15,
            )
