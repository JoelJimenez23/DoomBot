# doom_env/core/action_space.py

TURN_IDS          = {1, 2, 16, 17}
STRAFE_IDS        = {3, 4}
FWD_IDS           = {5, 16, 17}
BACK_IDS          = {6}

ATTACK_IDS        = {7, 8, 9, 10, 11, 12, 13}

TURN_ATTACK_IDS   = {8, 9}
STRAFE_ATTACK_IDS = {10, 11}
FWD_ATTACK_IDS    = {12}
BACK_ATTACK_IDS   = {13}
BACK_TURN_IDS     = {14, 15}

# útiles si quieres lógica/penalizaciones específicas
FWD_TURN_IDS      = {16, 17}


def actions_family_a():
    # Formato: [TL, TR, SL, SR, FWD, BACK, ATK]
    actions = [
        [False, False, False, False, False, False, False],  # 0 noop
        [True,  False, False, False, False, False, False],  # 1 turn_left
        [False, True,  False, False, False, False, False],  # 2 turn_right
        [False, False, True,  False, False, False, False],  # 3 strafe_left
        [False, False, False, True,  False, False, False],  # 4 strafe_right
        [False, False, False, False, True,  False, False],  # 5 forward
        [False, False, False, False, False, True,  False],  # 6 backward
        [False, False, False, False, False, False, True ],  # 7 attack
        [True,  False, False, False, False, False, True ],  # 8  turn_left + attack
        [False, True,  False, False, False, False, True ],  # 9  turn_right + attack
        [False, False, True,  False, False, False, True ],  # 10 strafe_left + attack
        [False, False, False, True,  False, False, True ],  # 11 strafe_right + attack
        [False, False, False, False, True,  False, True ],  # 12 forward + attack
        [False, False, False, False, False, True,  True ],  # 13 backward + attack
        [True,  False, False, False, False, True,  False],  # 14 back + turn_left
        [False, True,  False, False, False, True,  False],  # 15 back + turn_right
    ]
    return actions


def actions_family_b(nav_v2: bool = False):
    """
    Familia B: navegación/pickups.
    Formato: [TL, TR, SL, SR, FWD, BACK, ATK]
    Importante: ATK siempre False.
    """
    actions = [
        [False, False, False, False, False, False, False],  # 0 noop
        [True,  False, False, False, False, False, False],  # 1 turn_left
        [False, True,  False, False, False, False, False],  # 2 turn_right
        [False, False, False, False, True,  False, False],  # 3 forward
        # después de forward
        [True,  False, False, False, True,  False, False],  # turn_left + forward
        [False, True,  False, False, True,  False, False],  # turn_right + forward
    ]

    if nav_v2:
        # añade strafe si lo necesitas para maps tipo my_way_home
        actions += [
            [False, False, True,  False, False, False, False],  # 4 strafe_left
            [False, False, False, True,  False, False, False],  # 5 strafe_right
        ]

    return actions


def actions_family_ab(nav_v2: bool = False):
    """
    Merge A ∪ B manteniendo IDs de A intactos.

    - IDs 0..15: exactamente familia A (compatibilidad con tu lógica actual)
    - IDs 16..17: acciones extra de B (turn + forward) sin attack

    Esto permite health_gathering (navegación fluida) y mapas con combate.
    """
    actions = actions_family_a()

    # Acciones que B aporta y A no tiene:
    # turn_left + forward, turn_right + forward
    actions += [
        [True,  False, False, False, True,  False, False],  # 16 TL + FWD
        [False, True,  False, False, True,  False, False],  # 17 TR + FWD
    ]

    # nav_v2 (strafe) YA existe en A, así que no agregamos nada aquí.
    # Si quisieras strafe+forward, recién ahí habría que ampliar más.

    return actions


# IDs útiles para B (si quieres métricas/penalizaciones)
NAV_TURN_IDS = {1, 2}
NAV_FWD_IDS  = {3}
NAV_STRAFE_IDS = {4, 5}  # solo si nav_v2=True