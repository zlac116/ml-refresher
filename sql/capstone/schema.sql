-- Marketplace Seller Performance — schema + seed data.
-- PROVIDED scaffolding: load this as-is, do NOT edit. You write report.sql.

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS sellers;

CREATE TABLE sellers (
    id        smallint PRIMARY KEY,
    name      varchar(255) NOT NULL,
    category  varchar(255),            -- may be the literal 'n/a'
    gmv_tier  varchar(255)             -- e.g. '$2M', '$500K', '$1.5M', or 'n/a'
);

CREATE TABLE orders (
    seller_id   smallint     NOT NULL REFERENCES sellers(id),
    ordered_at  varchar(19)  NOT NULL,   -- 'YYYY-MM-DD HH:MI:SS'
    net_amount  numeric(10,2) NOT NULL   -- positive = sale, negative = refund/chargeback
);

INSERT INTO sellers (id, name, category, gmv_tier) VALUES
    (1, 'AlphaTech',  'Electronics', '$2M'),
    (2, 'BetaGoods',  'Electronics', '$500K'),
    (3, 'GammaWear',  'Apparel',     '$1.5M'),
    (4, 'DeltaHome',  'Apparel',     'n/a'),
    (5, 'EpsilonArt', 'n/a',         '$300K'),
    (6, 'ZetaToys',   'Toys',        '$50K');   -- no orders: exercises LEFT JOIN / 'none' cases

INSERT INTO orders (seller_id, ordered_at, net_amount) VALUES
    -- AlphaTech (Electronics)
    (1, '2025-02-10 09:00:00',  1000.00),   -- Q1
    (1, '2025-05-04 14:30:00',  2000.00),   -- Q2
    (1, '2024-12-20 11:00:00',  9999.00),   -- 2024 -> must be EXCLUDED
    -- BetaGoods (Electronics)
    (2, '2025-04-18 16:00:00',  -500.00),   -- Q2 refund
    (2, '2025-08-09 10:15:00',  -800.00),   -- Q3 refund (makes Q3 net-negative)
    (2, '2025-11-22 13:45:00',  1500.00),   -- Q4
    -- GammaWear (Apparel)
    (3, '2025-03-30 08:00:00',  3000.00),   -- Q1
    (3, '2025-07-12 19:20:00',  1000.00),   -- Q3
    -- DeltaHome (Apparel)
    (4, '2025-12-01 12:00:00',   500.00),   -- Q4
    -- EpsilonArt (n/a category -> "Other")
    (5, '2025-06-15 17:00:00',  2000.00);   -- Q2
