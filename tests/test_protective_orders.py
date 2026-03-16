from hf.execution.protective_orders import classify_reduce_orders, should_keep_partial_tps


def test_long_full_stop_and_two_partial_tps_above_ref():
    orders = [
        {"side": "sell", "amount": 232.0, "triggerPrice": 0.29627, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 116.0, "triggerPrice": 0.29882, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 116.0, "triggerPrice": 0.29991, "info": {"tradeSide": "close"}},
    ]

    stop_like, tp_like = classify_reduce_orders(
        orders,
        pos_side="long",
        ref_price=0.29950,
        pos_qty=232.0,
    )

    assert len(stop_like) == 1
    assert len(tp_like) == 2
    assert should_keep_partial_tps(tp_like, "long", 232.0) is True


def test_long_missing_one_tp_should_not_keep():
    orders = [
        {"side": "sell", "amount": 232.0, "triggerPrice": 0.29627, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 116.0, "triggerPrice": 0.29882, "info": {"tradeSide": "close"}},
    ]

    stop_like, tp_like = classify_reduce_orders(
        orders,
        pos_side="long",
        ref_price=0.29950,
        pos_qty=232.0,
    )

    assert len(stop_like) == 1
    assert len(tp_like) == 1
    assert should_keep_partial_tps(tp_like, "long", 232.0) is False


def test_long_three_tps_should_not_keep():
    orders = [
        {"side": "sell", "amount": 232.0, "triggerPrice": 0.29627, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 80.0, "triggerPrice": 0.29850, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 80.0, "triggerPrice": 0.29900, "info": {"tradeSide": "close"}},
        {"side": "sell", "amount": 72.0, "triggerPrice": 0.29950, "info": {"tradeSide": "close"}},
    ]

    stop_like, tp_like = classify_reduce_orders(
        orders,
        pos_side="long",
        ref_price=0.29960,
        pos_qty=232.0,
    )

    assert len(stop_like) == 1
    assert len(tp_like) == 3
    assert should_keep_partial_tps(tp_like, "long", 232.0) is False
